import torch

torch.set_default_dtype(torch.float64)

import gpytorch
from npcgp.utils import (
    ls2pr,
    pr2ls,
)
from npcgp.integrals import I_interdom
from math import pi


class FilterGP(torch.nn.Module):
    """
    Base approx. single input/output non-parametric GP class, which efficiently
    samples as per Wilson et al. [2020]. Used for the filter processes in our model.
    Uses DSE kernel with regular inducing points and no whitening.
    """

    def __init__(
        self,
        init_inducing_inputs,
        mc_samples=11,
        init_lengthscale=1.0,
        num_basis_functions=50,
        init_noise=0.001,
        scale_inputs=False,
        prior_cov_factor=1.0,
        prior_mean_factor=0.5,
        jitter=1e-7,
        device="cpu",
    ):
        super(FilterGP, self).__init__()
        self.d_in = init_inducing_inputs.shape[1]
        self.num_inducing_points = init_inducing_inputs.shape[0]
        self.num_basis_functions = num_basis_functions
        self.device = device
        self.mc_samples = mc_samples
        self.jitter = jitter
        self.raw_inducing_inputs = init_inducing_inputs

        if scale_inputs:
            self.register_parameter(
                "input_scale",
                torch.nn.Parameter(torch.tensor([1.0] * self.d_in, device=device)),
            )
        else:
            self.input_scale = torch.tensor([1.0] * self.d_in, device=device)

        self.register_parameter(
            "noise", torch.nn.Parameter(torch.tensor([init_noise], device=device))
        )

        self.kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.d_in, device=device)

        if type(init_lengthscale) is list:
            assert len(init_lengthscale) == self.d_in
            self.kernel.lengthscale = torch.tensor([init_lengthscale], device=device)
        else:
            self.kernel.lengthscale = torch.tensor(
                [[init_lengthscale] * self.d_in], device=device
            )

        self.kernel.to(device)

        self.variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
            self.num_inducing_points,
            device=device,
        )
        self.variational_dist.variational_mean = torch.nn.Parameter(
            prior_mean_factor * torch.ones(self.num_inducing_points, device=device)
        )

        self.variational_dist.chol_variational_covar = torch.nn.Parameter(
            prior_cov_factor * self.variational_dist.chol_variational_covar
        )
        self.variational_dist.to(device)

    @property
    def inducing_inputs(self):
        """Update and return inducing points."""
        return self.raw_inducing_inputs * self.input_scale[None, :]

    @property
    def alpha(self):
        """Update and return alpha parameter of DSE kernel."""
        return 3 / torch.max(torch.abs(self.inducing_inputs), 0)[0] ** 2

    @property
    def lengthscale(self):
        """Return lengthscale of DSE kernel."""
        return self.kernel.lengthscale[0]

    @property
    def prior(self):
        """Update and return prior distribution."""
        mean = torch.zeros(self.num_inducing_points, device=self.device)
        cov = self.kernel.forward(
            self.inducing_inputs, self.inducing_inputs
        ) + self.jitter * torch.eye(
            self.num_inducing_points, requires_grad=False, device=self.device
        )
        return gpytorch.distributions.MultivariateNormal(mean, cov, validate_args=True)

    def sample_basis(self):
        """Sample RFF basis."""
        thets = (
            torch.randn(
                self.mc_samples,
                self.d_in,
                self.num_basis_functions,
                requires_grad=False,
                device=self.device,
            )
            / self.lengthscale[None, :, None]
        )
        ws = torch.sqrt(
            torch.tensor(
                2.0 / self.num_basis_functions, requires_grad=False, device=self.device
            )
        ) * torch.randn(
            self.mc_samples,
            self.num_basis_functions,
            requires_grad=False,
            device=self.device,
        )
        betas = (
            2
            * pi
            * torch.rand(
                self.mc_samples,
                self.num_basis_functions,
                requires_grad=False,
                device=self.device,
            )
        )
        return thets, betas, ws

    def compute_q(self, basis):
        """Compute q matrix used in sampling computations as per Wilson et al. (2020)."""
        thets, betas, ws = basis
        phiz = torch.cos(self.inducing_inputs.matmul(thets) + betas[:, None, :])

        LKzz = torch.linalg.cholesky(self.prior.covariance_matrix)
        rLKzz = LKzz.unsqueeze(0).repeat(self.mc_samples, 1, 1)

        us = self.variational_dist.forward().rsample(
            sample_shape=torch.Size([self.mc_samples])
        )

        x = us[:, :, None] - phiz.matmul(ws[:, :, None])
        return torch.cholesky_solve(x, rLKzz).squeeze(-1)

    def forward(self, t):
        """Pass inputs through the GP."""
        if len(t.shape) == 2:
            t = t[None, :, :]
        basis = self.sample_basis()
        thets, betas, ws = basis
        phit = torch.cos(torch.einsum("btd, bdn ->btn", t, thets) + betas[:, None, :])
        qs = self.compute_q(basis)
        Ktz = self.kernel.forward(t, self.inducing_inputs)
        basis_part = torch.einsum("bz, btz -> bt", qs, Ktz)
        random_part = torch.einsum("bn, btn -> bt", ws, phit)
        return basis_part + random_part

    def compute_KL(self):
        """Compute KL divergence for use in ELBO."""
        kl = torch.distributions.kl.kl_divergence(
            self.variational_dist.forward(), self.prior
        )
        return kl

    def objective(self, x, y):
        """Computes ELBO using LL and KL divergence."""
        KL = self.compute_KL()
        samps = self.forward(x)
        likelihood = torch.distributions.Normal(y, 0.01)
        return KL - torch.mean(likelihood.log_prob(samps))

    def train(self, data, N_steps, lr):
        """Optimise model using ELBO for N_steps iterations."""
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for i in range(N_steps):
            opt.zero_grad()
            obj = self.objective(*data)
            obj.backward()
            opt.step()
            print(f"it {i}, obj {obj.item()}")


class InterDomainInputGP(torch.nn.Module):
    """
    Base approx. multi-input/output GP class with inter-domain transform, which efficiently
    samples as per Wilson et al. [2020]. Used for the input process in our model. Uses
    EQ kernel with optimised interdomain inducing points and whitening.
    """

    def __init__(
        self,
        init_inducing_inputs,
        inducing_outputs=None,
        mc_samples=11,
        process_lengthscale=0.1,
        transform_lengthscale=1.0,
        num_basis_functions=50,
        init_noise=0.001,
        prior_cov_factor=1.0,
        jitter=1e-7,
        device="cpu",
        whiten=True,
        num_gps=1,
    ):
        super(InterDomainInputGP, self).__init__()
        self.d_in = init_inducing_inputs.shape[1]
        self.num_inducing_points = init_inducing_inputs.shape[0]
        self.num_basis_functions = num_basis_functions
        self.device = device
        self.mc_samples = mc_samples
        self.whiten = whiten
        self.jitter = jitter
        self.num_gps = num_gps
        self.inducing_outputs = inducing_outputs

        self.register_parameter(
            "inducing_inputs", torch.nn.Parameter(init_inducing_inputs)
        )

        self.register_parameter(
            "noise", torch.nn.Parameter(torch.tensor([init_noise], device=device))
        )

        # Initialise transform lengthscales
        self.transform_lengthscale_constraint = gpytorch.constraints.LessThan(0.8)
        if type(transform_lengthscale) is list:
            self.register_parameter(
                "raw_transform_lengthscale",
                torch.nn.Parameter(
                    self.transform_lengthscale_constraint.inverse_transform(
                        torch.tensor(transform_lengthscale, device=device)
                    )
                ),
            )
        else:
            self.register_parameter(
                "raw_transform_lengthscale",
                torch.nn.Parameter(
                    self.transform_lengthscale_constraint.inverse_transform(
                        torch.tensor([transform_lengthscale] * self.d_in, device=device)
                    )
                ),
            )

        self.register_parameter(
            "transform_amp",
            torch.nn.Parameter(torch.tensor(1.0, device=device)),
        )

        # Initialise process lengthscales
        self.process_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=self.d_in,
            device=device,
            lengthscale_constraint=gpytorch.constraints.GreaterThan(0.1),
        )
        if type(process_lengthscale) is list:
            assert len(process_lengthscale) == self.d_in
            self.process_kernel.lengthscale = torch.tensor(
                [process_lengthscale], device=device
            )
        else:
            self.process_kernel.lengthscale = torch.tensor(
                [process_lengthscale] * self.d_in, device=device
            )

        self.process_kernel.to(device)

        # Initialise interdomain kernel
        self._interdomain_kernel_base = gpytorch.kernels.RBFKernel(
            ard_num_dims=self.d_in, device=device
        )
        _ = self.interdomain_kernel

        # Initialise cross kernel
        self._cross_kernel_base = gpytorch.kernels.RBFKernel(
            ard_num_dims=self.d_in, device=device
        )
        _ = self.cross_kernel

        self._cross_kernel_base.to(device)
        self._interdomain_kernel_base.to(device)

        # Initialise variational distribution
        batch_shape = torch.Size([num_gps])
        self.variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
            self.num_inducing_points, device=device, batch_shape=batch_shape
        )
        self.variational_dist.chol_variational_covar = torch.nn.Parameter(
            prior_cov_factor * self.variational_dist.chol_variational_covar
        )

        with torch.no_grad():
            s = self.prior.rsample(batch_shape)
            self.variational_dist.variational_mean = torch.nn.Parameter(s)

        self.variational_dist.to(device)

    @property
    def process_lengthscale(self):
        """Return lengthscale of untransformed process."""
        return self.process_kernel.lengthscale[0]

    @property
    def transform_lengthscale(self):
        """Return transform lengthscale."""
        return self.transform_lengthscale_constraint.transform(
            self.raw_transform_lengthscale
        )

    @property
    def interdomain_kernel(self):
        """Update and return the interdomain kernel."""
        ls_f = self.process_lengthscale
        ls_g = self.transform_lengthscale
        ls_u = pr2ls(1 / (1 / ls2pr(ls_f) + 2 / ls2pr(ls_g)))
        # print(ls_f, ls_g, ls_u)
        self._interdomain_kernel_base.lengthscale = ls_u
        return self._interdomain_kernel_base

    @property
    def interdomain_lengthscale(self):
        """Return the interdomain lengthscale."""
        return self.interdomain_kernel.lengthscale[0]

    @property
    def interdomain_amp(self):
        """Update and return the interdomain amplitude."""
        ls_f = self.process_lengthscale
        ls_g = self.transform_lengthscale
        a_u = torch.sqrt(
            torch.prod(ls2pr(ls_g) + ls2pr(ls_f))
            / torch.sqrt(torch.prod(ls2pr(ls_g) * (ls2pr(ls_g) + 2 * ls2pr(ls_f))))
        )
        return self.transform_amp**2 * a_u

    @property
    def cross_kernel(self):
        """Update and return kernel which defines the cross-covariance."""
        ls_f = self.process_lengthscale
        ls_g = self.transform_lengthscale
        ls_fu = pr2ls(1 / (1 / ls2pr(ls_f) + 1 / ls2pr(ls_g)))
        self._cross_kernel_base.lengthscale = ls_fu
        return self._cross_kernel_base

    @property
    def cross_lengthscale(self):
        """Return the cross kernel lengthscale."""
        return self.cross_kernel.lengthscale[0]

    @property
    def cross_amp(self):
        """Return the cross kernel amplitude."""
        return self.transform_amp

    @property
    def prior(self):
        """Update and return prior distribution."""
        mean = torch.zeros(self.num_inducing_points, device=self.device)
        cov = self.interdomain_amp * (
            self.interdomain_kernel.forward(self.inducing_inputs, self.inducing_inputs)
            + self.jitter
            * torch.eye(
                self.num_inducing_points, requires_grad=False, device=self.device
            )
        )
        return gpytorch.distributions.MultivariateNormal(mean, cov, validate_args=True)

    def sample_basis(self):
        "Sample RFF basis."
        thets = (
            torch.randn(
                self.mc_samples,
                self.num_gps,
                self.d_in,
                self.num_basis_functions,
                requires_grad=False,
                device=self.device,
            )
            / self.process_lengthscale[None, None, :, None]
        )
        ws = torch.sqrt(
            torch.tensor(
                2.0 / self.num_basis_functions, requires_grad=False, device=self.device
            )
        ) * torch.randn(
            self.mc_samples,
            self.num_gps,
            self.num_basis_functions,
            requires_grad=False,
            device=self.device,
        )
        betas = (
            2
            * pi
            * torch.rand(
                self.mc_samples,
                self.num_gps,
                self.num_basis_functions,
                requires_grad=False,
                device=self.device,
            )
        )
        return thets, betas, ws

    def compute_q(self, basis):
        """Compute q matrix used in sampling computations as per Wilson et al. (2020)."""
        thets, betas, ws = basis
        phiz = I_interdom(
            self.inducing_inputs, ls2pr(self.transform_lengthscale), thets, betas
        )

        LKzz = torch.linalg.cholesky(self.prior.covariance_matrix)

        if self.inducing_outputs is None:
            us = self.variational_dist.forward().rsample(
                sample_shape=torch.Size([self.mc_samples])
            )
        else:
            us = self.inducing_outputs[None, None, :]

        if self.whiten:
            us = us.matmul(LKzz)

        x = us[:, :, :, None] - phiz.matmul(ws[:, :, :, None])
        return torch.cholesky_solve(x, LKzz[None, None, :, :]).squeeze(-1)

    def forward(self, t):
        "Pass inputs through GP."
        if len(t.shape) == 2:
            t = t[None, :, :]
        basis = self.sample_basis()
        thets, betas, ws = basis
        phit = torch.cos(
            torch.einsum("btd, bqdn ->bqtn", t, thets) + betas[:, :, None, :]
        )
        qs = self.compute_q(basis)
        Ktz = self.cross_amp * self.cross_kernel.forward(t, self.inducing_inputs)
        basis_part = torch.einsum("bqz, btz -> bqt", qs, Ktz)
        random_part = torch.einsum("bqn, bqtn -> bqt", ws, phit)
        return basis_part + random_part

    def compute_KL(self):
        """Compute KL divergence for use in ELBO."""
        kl = sum(
            [
                torch.distributions.kl.kl_divergence(disti, self.prior)
                for disti in self.variational_dist.forward()
            ]
        )
        return kl

    def objective(self, x, y):
        "Computes ELBO using LL and KL divergence."
        KL = self.compute_KL()
        samps = self.forward(x)
        likelihood = torch.distributions.Normal(y, self.noise)
        return KL - torch.mean(likelihood.log_prob(samps))

    def train(self, data, N_steps, lr):
        "Optimise model using ELBO for N_steps iterations."
        opt_param_names = []
        pars = dict(self.named_parameters())
        for p in list(pars):
            if ("interdomain" in p) or ("cross" in p):
                pars.pop(p, None)
            else:
                opt_param_names.append(p)
        print("Optimised parameters:", opt_param_names)
        opt = torch.optim.Adam(pars.values(), lr=lr)
        for i in range(N_steps):
            opt.zero_grad()
            obj = self.objective(*data)
            obj.backward()
            opt.step()
            print(f"it {i}, obj {obj.item()}")
