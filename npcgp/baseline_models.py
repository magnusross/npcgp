import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    """
    Multitask SVGP model, based on implementation available at:
    https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html
    """

    def __init__(self, inducing_points, num_latents, num_tasks, device):
        # Let's use a different set of inducing points for each latent function
        inducing_points = inducing_points[None, :, :].repeat(num_latents, 1, 1)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        ).to(device)

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        ).to(device)

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents])
        ).to(device)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]),
        ).to(device)

    def forward(self, x):
        """Pass inputs through GP."""
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactMultitaskGPModel(gpytorch.models.ExactGP):
    """
    Multitask exact GP model, based on implementation available at:
    "https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html"
    """

    def __init__(self, train_x, train_y, likelihood, num_tasks, device):
        super(ExactMultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        ).to(device)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        ).to(device)

    def forward(self, x):
        """Pass inputs through GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def fit_gpytorch_mogp(
    X_train,
    y_train,
    X_valid,
    y_valid,
    inducing_points,
    n_iter=300,
    num_latents=3,
    num_tasks=4,
    device="cpu",
    variational=True,
):
    """Convenience function which fits both of the above MO baseline models on given data."""
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks
    ).to(device)

    if variational:
        model = MultitaskGPModel(
            inducing_points=inducing_points,
            num_latents=num_latents,
            num_tasks=num_tasks,
            device=device,
        )
    else:
        model = ExactMultitaskGPModel(X_train, y_train, likelihood, num_tasks, device)

    model.train()
    likelihood.train()

    if variational:
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": likelihood.parameters()},
            ],
            lr=1e-3,
        )
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

        training = True
        iter = 0
        print("Fitting GPyTorch model...")
        while training:
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                iter += 1
                if (iter % 100) == 0:
                    print("Iteration %d complete." % iter)

                if iter >= n_iter:
                    training = False
                    break
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-3
        )  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        iter = 0
        print("Fitting GPyTorch model...")
        for i in range(n_iter):
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            iter += 1
            if (iter % 100) == 0:
                print("Iteration %d complete." % iter)

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Make validation set predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = model(X_valid)
        predictions = likelihood(output)

    return predictions, likelihood.noise
