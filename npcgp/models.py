import torch

torch.set_default_dtype(torch.float64)

import time
import wandb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from npcgp import integrals
from npcgp.base_gps import FilterGP, InterDomainInputGP
from npcgp.utils import (
    ls2pr,
    pr2ls,
    to_numpy,
    double_integral,
    batch_assess,
)
from math import pi, sqrt, log, ceil


class NPCGP(torch.nn.Module):
    """
    Implements a nonparametric multi-output GP, which accepts a multivariate input.
    Here, each input/output are assigned their own filter IPs, with the input IPs shared
    across the input dimensions.
    """

    def __init__(
        self,
        X,
        num_outputs,
        init_u_inducing_inputs,
        init_filter_width=1.0,
        init_transform_lengthscale=1.5,
        init_amp=1.0,
        init_noise=0.01,
        num_u_functions=None,
        num_filter_points=10,
        mc_samples=10,
        device="cpu",
        beta=0.5,
        prior_cov_factor_u=1.0,
        prior_cov_factor_g=0.5,
        prior_mean_factor_g=0.5,
        **kwargs,
    ):
        super(NPCGP, self).__init__()
        self.N_data = X.shape[0]
        self.d_out = num_outputs
        self.d_in = init_u_inducing_inputs.shape[1]
        self.num_u_inducing_points = init_u_inducing_inputs.shape[0]
        self.num_filter_points = num_filter_points
        self.device = device
        if num_u_functions is None:
            num_u_functions = self.d_out
        self.num_u_functions = num_u_functions
        self.init_u_inducing_inputs = init_u_inducing_inputs
        self.mc_samples = mc_samples

        assert self.d_in == X.shape[1]

        self.register_parameter(
            "log_noise",
            torch.nn.Parameter(
                torch.log(torch.tensor([init_noise] * self.d_out, device=device))
            ),
        )

        # Create initial inducing inputs and lengthscales for all filters
        if type(init_filter_width) is list:
            assert len(init_filter_width) == self.d_in
        else:
            init_filter_width = [init_filter_width] * self.d_in

        self.init_g_inducing_inputs = [
            init_filter_width[i]
            * torch.linspace(
                -1, 1, num_filter_points, requires_grad=False, device=device
            ).reshape(-1, 1)
            for i in range(self.d_in)
        ]
        self.init_g_lengthscale = [
            (
                1.0
                * (
                    self.init_g_inducing_inputs[i][1, 0]
                    - self.init_g_inducing_inputs[i][0, 0]
                ).item()
            )
            for i in range(self.d_in)
        ]
        self.init_u_lengthscale = [ls * beta for ls in self.init_g_lengthscale]
        self.init_transform_lengthscale = init_transform_lengthscale

        self.set_gps(
            prior_mean_factor_g, prior_cov_factor_g, prior_cov_factor_u, **kwargs
        )
        try:  # for full
            init_alpha = self.g_gps[0][0].alpha
        except TypeError:  # for fast
            init_alpha = self.g_gps[0].alpha

        normaliser = (2 * init_alpha / pi) ** 0.25
        self.register_parameter(
            "log_amps",
            torch.nn.Parameter(
                log(init_amp)
                + torch.log(normaliser)
                + torch.zeros((self.d_out, self.num_u_functions), device=device)
            ),
        )

    @property
    def amps(self):
        """Return amplitude parameters."""
        return torch.exp(self.log_amps)

    def set_gps(
        self, prior_mean_factor_g, prior_cov_factor_g, prior_cov_factor_u, **kwargs
    ):
        """Build model by constructing array of filter GPs for G and the input process u."""
        self.g_gps = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        FilterGP(
                            self.init_g_inducing_inputs[p],
                            init_lengthscale=self.init_g_lengthscale[p],
                            device=self.device,
                            scale_inputs=True,
                            prior_cov_factor=prior_cov_factor_g,
                            prior_mean_factor=prior_mean_factor_g,
                            **kwargs,
                        )
                        for p in range(self.d_in)
                    ]
                )
                for d in range(self.d_out)
            ]
        )
        self.u_gp = InterDomainInputGP(
            self.init_u_inducing_inputs,
            process_lengthscale=self.init_u_lengthscale,
            transform_lengthscale=self.init_transform_lengthscale,
            device=self.device,
            prior_cov_factor=prior_cov_factor_u,
            num_gps=self.num_u_functions,
            **kwargs,
        )

    def _gs_list2torch(self, d_idx, attr):
        """Convenience function to convert list of filter GPs to tensor."""
        g_gps_d = self.g_gps[d_idx]
        return torch.hstack([getattr(gp, attr) for gp in g_gps_d])

    def _gs_sample_basis(self, d_idx):
        """Sample RFF basis for all filter GPs."""
        thets = []
        betas = []
        ws = []
        qs = []
        for gp in self.g_gps[d_idx]:
            basis = gp.sample_basis()
            thetd, betasd, wsd = basis
            thets.append(thetd)
            betas.append(betasd)
            ws.append(wsd)
            qs.append(gp.compute_q(basis))
        return (
            torch.hstack(thets),
            torch.swapaxes(torch.stack(betas), 0, 1),
            torch.swapaxes(torch.stack(ws), 0, 1),
            torch.swapaxes(torch.stack(qs), 0, 1),
        )

    def integrate_covariance(self, g_gp, pu, dts, N_tau=100, Ns=5):
        """Compute integral required to plot the covariance."""
        pg = ls2pr(g_gp.lengthscale[0]).item()

        max_tau = max_tau = 3 / sqrt(
            ((g_gp.alpha + 2 * pg) * pu) / (g_gp.alpha + 2 * (pg + pu))
        )
        taus = torch.linspace(-3 * max_tau, 3 * max_tau, N_tau, device=self.device)
        ta, tap = torch.meshgrid((taus, taus), indexing="ij")
        kt = torch.zeros(Ns, len(dts), device=self.device)
        for k in range(Ns):
            t = dts[:, None, None]
            tp = torch.zeros((len(dts), 1, 1), device=self.device)
            tau = ta.reshape(1, -1, 1)
            taup = tap.reshape(1, -1, 1)
            norm_part = torch.exp(
                -g_gp.alpha * ((tau) ** 2 + (taup) ** 2)
                - pu * ((t - tau) - (tp - taup)) ** 2
            )
            random_part = g_gp.forward(torch.cat((tau, taup), axis=1).reshape(-1, 1))[0]
            fxy = (
                norm_part[:, :, 0]
                * random_part[None, : tau.shape[1]]
                * random_part[None, tau.shape[1] :]
            ).reshape(-1, N_tau, N_tau)

            kt[k] = double_integral(
                -max_tau, max_tau, -max_tau, max_tau, N_tau, N_tau, fxy
            )
        return kt

    def sample_covariance(self, Ns=5):
        """Use covariance integral to sample and generate data for plotting."""
        if self.d_out > 1:
            raise ValueError("Function only works for single output models.")

        with torch.no_grad():
            out = []
            out_dts = []
            N_tau = 100

            for i, gps in enumerate(self.g_gps):
                outi = []
                out_dtsi = []
                for j, gp in enumerate(gps):
                    u_pr = ls2pr(self.u_gp.process_lengthscale[j]).item()
                    max_a = 1.5 * 3 / sqrt(gp.alpha)
                    max_u = 1.5 * 3 / sqrt(u_pr)
                    max_dt = max(max_a, max_u)
                    dts = torch.linspace(0, max_dt, 100, device=self.device)
                    kt = self.integrate_covariance(gp, u_pr, dts, Ns=Ns)
                    outi.append(kt * self.amps[0][0] ** 2)
                    out_dtsi.append(dts)
                out.append(outi)
                out_dts.append(out_dtsi)
        return (out_dts, out)

    def sample_filter(self, tg=None):
        """Draw samples from filter GPs."""
        with torch.no_grad():
            out_ts = []
            out_fs = []
            for i, gps in enumerate(self.g_gps):
                out_tsi = []
                out_fsi = []
                for j, gp in enumerate(gps):
                    max_ip = torch.max(torch.abs(gp.inducing_inputs)).item()
                    tg = torch.linspace(
                        -max_ip, max_ip, 300, requires_grad=False, device=self.device
                    ).reshape(-1, 1)
                    fs = torch.exp(-gp.alpha * tg**2).T * gp.forward(tg)
                    out_fsi.append(fs * torch.sum(self.amps[i]))
                    out_tsi.append(tg)
                out_ts.append(out_tsi)
                out_fs.append(out_fsi)

        return out_ts, out_fs

    def forward(self, ts):
        """Pass inputs through the NP-CGP."""
        u_basis = self.u_gp.sample_basis()
        thetaus, betaus, wus = u_basis
        qus = self.u_gp.compute_q(u_basis)
        pus = ls2pr(self.u_gp.cross_lengthscale)
        zus = self.u_gp.inducing_inputs
        thetaus = torch.swapaxes(thetaus, 2, 3)
        ampu = self.u_gp.cross_amp
        out = []

        for i in range(self.d_out):

            pgs = ls2pr(self._gs_list2torch(i, "lengthscale"))
            alphas = self._gs_list2torch(i, "alpha")
            zgs = self._gs_list2torch(i, "inducing_inputs")
            thetags, betags, wgs, qgs = self._gs_sample_basis(i)

            outi = integrals.full_I(
                ts,
                alphas,
                pgs,
                wgs,
                thetags,
                betags,
                zgs,
                qgs,
                pus,
                wus,
                thetaus,
                betaus,
                zus,
                qus,
                ampu,
            )
            out.append(outi)

        layer_out = torch.stack(out, -1)
        layer_out = (self.amps.T[None, :, None, :] * layer_out).sum(axis=1)

        return layer_out

    def forward_multiple_mc(self, t, S=100):
        """
        Allows for samples to be generated during evaluation
        with mc_multiplier times more MC samples than used during training.
        """
        if S < self.mc_samples:
            mc_multiplier = 1
        else:
            mc_multiplier = int(ceil(S / self.mc_samples))
        all_samps = None
        for i in range(mc_multiplier):
            with torch.no_grad():
                samps = self.forward(t)
                if all_samps is None:
                    all_samps = self.forward(t)
                else:
                    all_samps = torch.cat([all_samps, samps], 0)

        return all_samps[:S]

    def compute_KL(self):
        """
        Compute KL divergence by summing KL for input process with
        all of the KLs for the filter GPs.
        """
        kl = 0.0
        for g_d in self.g_gps:
            for g_d_p in g_d:
                kl += g_d_p.compute_KL()

        return kl + self.u_gp.compute_KL()

    def plot_features(self, save=None, covariances=True):
        """
        Plot the form of the filters. Setting `covariances=True`
        instead plots the form of the covariance.
        """
        with torch.no_grad():

            fig = plt.figure(
                constrained_layout=False,
                figsize=(self.d_in * 2, self.d_out * 2),
            )
            grsp = GridSpec(self.d_out, self.d_in, figure=fig)

            if covariances:
                tgs, g_samps = self.sample_covariance()
            else:
                tgs, g_samps = self.sample_filter()
            for j in range(self.d_out):
                for k in range(self.d_in):
                    g_ax = fig.add_subplot(grsp[j, k])
                    gm = torch.mean(g_samps[j][k], axis=0)
                    gs = torch.std(g_samps[j][k], axis=0)

                    ts = to_numpy(tgs[j][k]).flatten()
                    g_ax.plot(
                        ts,
                        to_numpy(gm),
                        c=plt.get_cmap("Set2")(j),
                    )
                    g_ax.fill_between(
                        ts,
                        to_numpy(gm) - to_numpy(gs),
                        to_numpy(gm) + to_numpy(gs),
                        color=plt.get_cmap("Set2")(j),
                        alpha=0.4,
                    )
                    g_ax.plot(
                        ts,
                        to_numpy(g_samps[j][k]).T,
                        color=plt.get_cmap("Set2")(j + 1),
                        alpha=0.2,
                    )
                    if not covariances:
                        g_ax.scatter(
                            to_numpy(self.g_gps[j][k].inducing_inputs),
                            to_numpy(
                                torch.exp(
                                    -self.g_gps[j][k].alpha
                                    * self.g_gps[j][k].inducing_inputs ** 2
                                )[:, 0]
                                * self.g_gps[j][k].variational_dist.variational_mean
                            ),
                            color=plt.get_cmap("Set2")(j),
                            alpha=0.7,
                        )
            plt.tight_layout()
            if save is not None:
                plt.savefig(
                    save,
                    dpi=300,
                    bbox_inches="tight",
                )
            else:
                plt.show()

    def plot_amps(self, save=None):
        """Visualise the amplitude parameters."""
        fig, ax = plt.subplots()
        arr = to_numpy(self.amps)
        ax.imshow(arr)
        for i in range(self.num_u_functions):
            for j in range(self.d_out):
                text = ax.text(
                    i, j, f"{arr[j, i]:.3f}", ha="center", va="center", color="w"
                )
        ax.set_xlabel("input functions")
        ax.set_ylabel("outputs")
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()

    def objective(self, x, y):
        """Compute ELBO using KL divergence and approximation of the LL."""
        KL = self.compute_KL()
        samps = self.forward(x)

        # If there are missing output observations, compute objective output by output
        if torch.isnan(y).any():
            like = 0.0
            for i in range(self.num_outputs):
                y_d = y[:, i]
                samps_d = samps[:, :, i]

                # Remove missing (NaN) values from target and samples
                is_nan = torch.isnan(y_d)
                y_d = y_d[~is_nan]
                samps_d = samps_d[:, ~is_nan]

                like += torch.sum(
                    torch.mean(
                        (
                            -0.5 * log(2 * pi)
                            - 0.5 * self.log_noise[i]
                            - 0.5
                            * ((y_d - samps_d) ** 2 / torch.exp(self.log_noise[i]))
                        ),
                        0,
                    )
                ) * (self.N_data / samps.shape[1])

        # Otherwise, compute likelihood all at once
        else:
            like = torch.sum(
                torch.mean(
                    (
                        -0.5 * log(2 * pi)
                        - 0.5 * self.log_noise[None, None, :]
                        - 0.5
                        * ((y - samps) ** 2 / torch.exp(self.log_noise[None, None, :]))
                    ),
                    0,
                )
            ) * (self.N_data / samps.shape[1])

        return KL - like

    def get_metrics(self, output, y, y_scale=1.0):
        """Return evaluation metrics."""
        metrics = {}
        y_pred_mean = torch.mean(output, 0)

        mnll = -torch.mean(
            torch.mean(
                (
                    -0.5 * log(2 * pi)
                    - 0.5
                    * torch.log(
                        torch.exp(self.log_noise[None, None, :])
                        * y_scale[None, None, :] ** 2
                    )
                    - 0.5
                    * (
                        (y * y_scale - output * y_scale) ** 2
                        / (
                            torch.exp(self.log_noise[None, None, :])
                            * y_scale[None, None, :] ** 2
                        )
                    )
                ),
                0,
            )
        )

        metrics["mnll"] = mnll.item()
        metrics["nmse"] = (
            torch.mean((y_pred_mean - y) ** 2) / torch.mean((torch.mean(y) - y) ** 2)
        ).item()
        metrics["mse"] = (torch.mean(y_scale**2 * (y_pred_mean - y) ** 2)).item()

        return metrics

    def predict(self, x, y=None, y_scale=1.0, S=None):
        """Return metrics and compute predictions for a given input."""
        with torch.no_grad():

            # Sample self.mc_samples
            if S is None:
                output = self.forward(x)
            # Sample S samples
            else:
                output = self.forward_multiple_mc(x, S=S)

            y_pred_mean = torch.mean(output, 0)
            y_pred_std = torch.std(output, 0)

            if y is not None:
                # If there are missing output observations, compute objective output by output
                if torch.isnan(y).any():
                    metrics_list = []
                    for i in range(self.num_outputs):
                        y_d = y[:, i]
                        output_d = output[:, :, i]

                        # Remove missing (NaN) values from target and samples
                        is_nan = torch.isnan(y_d)
                        y_d = y_d[~is_nan]
                        output_d = output_d[:, ~is_nan]
                        metrics_d = self.get_metrics(output_d, y_d, y_scale=y_scale)
                        metrics_list.append(metrics_d)

                        metrics = {}
                        # Average metrics across all outputs
                        for key in metrics_list[0].keys():
                            metrics[key] = sum(
                                [metrics_d[key] for metrics_d in metrics_list]
                            ) / len(metrics_list)

                        return y_pred_mean, y_pred_std, metrics

                # Otherwise, compute metrics all at once
                else:
                    metrics = self.get_metrics(output, y, y_scale=y_scale)
                    return y_pred_mean, y_pred_std, metrics
            else:
                return y_pred_mean, y_pred_std

    def predict_outputs(self, x, y=None, y_scale=1.0, S=None):
        """Compute predictions and return metrics output by output (not used in training)."""
        with torch.no_grad():

            # Sample self.mc_samples
            if S is None:
                output = self.forward(x)
            # Sample S samples
            else:
                output = self.forward_multiple_mc(x, S=S)

            y_pred_mean = torch.mean(output, 0)
            y_pred_std = torch.std(output, 0)

            # If there are missing output observations, compute objective output by output
            if y is not None:
                metrics_list = []
                for i in range(self.num_outputs):
                    y_d = y[:, i]
                    output_d = output[:, :, i]

                    # Remove missing (NaN) values from target and samples
                    if torch.isnan(y_d).any():
                        is_nan = torch.isnan(y_d)
                        y_d = y_d[~is_nan]
                        output_d = output_d[:, ~is_nan]
                    metrics_d = self.get_metrics(output_d, y_d, y_scale=y_scale)
                    metrics_list.append(metrics_d)

                metrics = {}
                # Store metrics for all outputs
                for key in metrics_list[0].keys():
                    metrics[key] = [
                        metrics_list[i][key] for i in range(self.num_outputs)
                    ]

                return y_pred_mean, y_pred_std, metrics

            else:
                return y_pred_mean, y_pred_std

    def eval_step(
        self,
        data,
        data_valid,
        y_scale,
        current_iter,
        steps_per_s,
        train_time,
        obj,
        batch_size,
    ):
        """Helper function which computes metrics for logging during training."""
        with torch.no_grad():
            X_train, y_train = data
            subset_size = min(self.N_data, batch_size)
            subset_idx = torch.randint(
                y_train.shape[0],
                size=(subset_size,),
                requires_grad=False,
                device=self.device,
            )
            train_metrics = batch_assess(
                self,
                X_train[subset_idx, :],
                y_train[subset_idx, :],
                device=self.device,
                y_scale=y_scale,
                S=5,
            )
            metrics = batch_assess(
                self,
                *data_valid,
                device=self.device,
                y_scale=y_scale,
                S=5,
            )
        print("Iteration %d" % (current_iter))
        print(
            "Steps/s = %.4f, Time = %.4f, Bound = %.4f, Train RMSE = %.4f, Validation RMSE = %.4f, Validation MNLL = %.4f\n"
            % (
                steps_per_s,
                train_time / 1000,
                obj.item(),
                train_metrics["rmse"],
                metrics["rmse"],
                metrics["mnll"],
            )
        )
        wandb.log(
            {
                "iter": current_iter,
                "train rmse": train_metrics["rmse"],
                "val rmse": metrics["rmse"],
                "val mnll": metrics["mnll"],
                "bound": obj.item(),
            }
        )

        return metrics, train_metrics

    def train(
        self,
        data,
        data_valid=None,
        n_iter=100,
        lr=1e-3,
        verbosity=1,
        batch_size=128,
        train_time_limit=None,
        y_scale=None,
        fix_g_pars=False,
        model_filepath="model.torch",
        chunks=1,
    ):
        """Train NP-CGP on supplied inputs."""
        train_time = 0

        # If no validation set specified, just evaluate on the training data
        if data_valid is None:
            data_valid = data

        # Set optimiser and parameters to be optimised
        pars = dict(self.named_parameters())
        for p in list(pars):
            if ("process_kernel.raw_lengthscale") in p:
                pars.pop(p, None)
            if fix_g_pars:
                if ("g_gps" in p) and not ("input" in p):
                    pars.pop(p, None)
        opt = torch.optim.Adam(pars.values(), lr=lr)

        # Initialise dataloader for minibatch training
        train_dataset = torch.utils.data.TensorDataset(*data)
        if chunks > 1:
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=int(batch_size / chunks), shuffle=True
            )
        else:
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

        times_list = []
        start_time = int(round(time.time() * 1000))
        steps_start_time = time.time()
        current_iter = 0
        step_counter = 0
        training = True

        # Perform iterations of minibatch training
        while training:

            if chunks > 1:
                chunk_counter = 0
                for X_minibatch, y_minibatch in train_dataloader:
                    obj = self.objective(
                        X_minibatch.to(self.device),
                        y_minibatch.to(self.device),
                    )
                    obj /= chunks
                    obj.backward()
                    chunk_counter += 1

                    if chunk_counter == chunks:
                        chunk_counter = 0
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        step_counter += 1
                        current_iter += 1

                        # Stop training after time limit elapsed
                        if train_time_limit is not None:
                            if train_time > 1000 * 60 * train_time_limit:
                                print("\nTraining Complete.\n")
                                training = False
                                break

                        # Stop training if specified number of iterations has been completed
                        if current_iter >= n_iter:
                            print("\nTraining Complete.\n")
                            training = False
                            break

                        # Display validation metrics at specified intervals
                        if verbosity == 0:
                            if current_iter % 100 == 0:
                                print(
                                    "%d training iterations completed." % current_iter
                                )
                        elif current_iter % verbosity == 0:
                            steps_per_s = step_counter / (
                                time.time() - steps_start_time
                            )

                            _, _ = self.eval_step(
                                (
                                    X_minibatch.to(self.device),
                                    y_minibatch.to(self.device),
                                ),  # NOTE - using this instead of `data` to save memory
                                data_valid,
                                y_scale,
                                current_iter,
                                steps_per_s,
                                train_time,
                                obj,
                                batch_size,
                            )
                            times_list.append(train_time / 1000)
                            step_counter = 0
                            steps_start_time = time.time()
                            train_time = int(round(time.time() * 1000)) - start_time

            else:
                # Use each batch to train model
                for X_minibatch, y_minibatch in train_dataloader:
                    obj = self.objective(
                        X_minibatch.to(self.device),
                        y_minibatch.to(self.device),
                    )
                    obj.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step_counter += 1
                    current_iter += 1

                    # Stop training after time limit elapsed
                    if train_time_limit is not None:
                        if train_time > 1000 * 60 * train_time_limit:
                            print("\nTraining Complete.\n")
                            training = False
                            break

                    # Stop training if specified number of iterations has been completed
                    if current_iter >= n_iter:
                        print("\nTraining Complete.\n")
                        training = False
                        break

                    # Display validation metrics at specified intervals
                    if verbosity == 0:
                        if current_iter % 100 == 0:
                            print("%d training iterations completed." % current_iter)
                    elif current_iter % verbosity == 0:
                        steps_per_s = step_counter / (time.time() - steps_start_time)

                        _, _ = self.eval_step(
                            (
                                X_minibatch.to(self.device),
                                y_minibatch.to(self.device),
                            ),  # NOTE - using this instead of `data` to save memory
                            data_valid,
                            y_scale,
                            current_iter,
                            steps_per_s,
                            train_time,
                            obj,
                            batch_size,
                        )
                        times_list.append(train_time / 1000)
                        step_counter = 0
                        steps_start_time = time.time()
                        train_time = int(round(time.time() * 1000)) - start_time

        # Save the final model once training complete
        torch.save(self.state_dict(), model_filepath)

    def plot_predictions(self, X, y, save=None, S=20):
        """
        Plots predictions; note that this is primarily useful for time series
        (e.g. rather than UCI data), but can be a useful tool still.
        """
        with torch.no_grad():
            m, std = self.predict(X, S=S)

            if X.shape < y.shape:
                y_vert = True
                fig, axs = plt.subplots(
                    y.shape[1], X.shape[1], figsize=(10, y.shape[1] * 3), squeeze=False
                )

            else:
                y_vert = False
                fig, axs = plt.subplots(
                    X.shape[1], y.shape[1], figsize=(10, 3 * X.shape[1]), squeeze=False
                )

            for j in range(y.shape[1]):
                for i in range(X.shape[1]):
                    sort_idx = torch.argsort(X[:, i])

                    mi = m[sort_idx, j].detach().cpu().numpy().flatten()
                    si = std[sort_idx, j].detach().cpu().numpy().flatten()
                    Xi = X[sort_idx, i].detach().cpu().numpy()
                    Zs = self.u_gp.inducing_inputs[:, i].detach().cpu().numpy()

                    if y_vert:
                        ax = axs[j, i]
                    else:
                        ax = axs[i, j]

                    ax.plot(Xi, mi)
                    ax.fill_between(Xi, mi - si, mi + si, alpha=0.25)

                    if y is not None:
                        yi = y[sort_idx, j].detach().cpu().numpy().flatten()
                        ax.scatter(Xi, yi, alpha=0.25)
                        ax.vlines(
                            Zs,
                            0,
                            1,
                            alpha=0.25,
                            color="red",
                            transform=ax.get_xaxis_transform(),
                        )

            if save is not None:
                plt.savefig(save)
            else:
                plt.show()

    def plot_output_slices(self, X, save=None):
        """Plots slices from the model, useful tool for visualising samples."""
        with torch.no_grad():
            dims = X.shape[1]
            xsi = torch.linspace(
                -torch.max(X),
                torch.max(X),
                500,
                requires_grad=False,
                device=self.device,
            )

            fig, axs = plt.subplots(dims, 1, figsize=(6, 2 * dims), squeeze=False)
            for i in range(dims):
                xl = [torch.zeros((500, 1), device=self.device)] * dims
                xl[i] = xsi[:, None]
                x_slice = torch.hstack(xl)
                out = self.forward(x_slice)
                axs[i][0].plot(to_numpy(xsi), to_numpy(out)[:, :, 0].T)

            if save is not None:
                plt.savefig(save)
            else:
                plt.show()


class FastNPCGP(NPCGP):
    """
    Fast variant of the NP-CGP described in the paper. This version of the model
    is only applicable in the multi-output setting, as when we have a single output
    it is functionally equivalent to the regular NP-CGP.
    """

    def set_gps(
        self, prior_mean_factor_g, prior_cov_factor_g, prior_cov_factor_u, **kwargs
    ):
        self.g_gps = torch.nn.ModuleList(
            [
                FilterGP(
                    self.init_g_inducing_inputs[p],
                    init_lengthscale=self.init_g_lengthscale[p],
                    device=self.device,
                    scale_inputs=True,
                    prior_cov_factor=prior_cov_factor_g,
                    prior_mean_factor=prior_mean_factor_g,
                    **kwargs,
                )
                for p in range(self.d_in)
            ]
        )

        self.u_gp = InterDomainInputGP(
            self.init_u_inducing_inputs,
            process_lengthscale=self.init_u_lengthscale,
            transform_lengthscale=self.init_transform_lengthscale,
            device=self.device,
            prior_cov_factor=prior_cov_factor_u,
            num_gps=self.num_u_functions,
            **kwargs,
        )

    def sample_covariance(self, Ns=5):
        with torch.no_grad():
            out = []
            out_dts = []
            N_tau = 100

            for i, gp in enumerate(self.g_gps):
                u_pr = ls2pr(self.u_gp.process_lengthscale[i]).item()
                max_a = 1.5 * 3 / sqrt(gp.alpha)
                max_u = 1.5 * 3 / sqrt(u_pr)
                max_dt = max(max_a, max_u)
                dts = torch.linspace(0, max_dt, 100, device=self.device)
                kt = self.integrate_covariance(gp, u_pr, dts)
                out.append(kt * torch.mean(self.amps))
                out_dts.append(dts)
        return (out_dts, out)

    def sample_filter(self, tg=None):
        with torch.no_grad():
            out_ts = []
            out_fs = []
            for i, gp in enumerate(self.g_gps):
                max_ip = torch.max(torch.abs(gp.inducing_inputs)).item()
                tg = torch.linspace(
                    -max_ip, max_ip, 300, requires_grad=False, device=self.device
                ).reshape(-1, 1)
                fs = torch.exp(-gp.alpha * tg**2).T * gp.forward(tg)
                out_fs.append(fs * torch.mean(self.amps))
                out_ts.append(tg)

        return out_ts, out_fs

    def forward(self, ts):

        u_basis = self.u_gp.sample_basis()
        thetaus, betaus, wus = u_basis
        qus = self.u_gp.compute_q(u_basis)
        pus = ls2pr(self.u_gp.cross_lengthscale)
        zus = self.u_gp.inducing_inputs
        thetaus = torch.swapaxes(thetaus, 2, 3)
        ampu = self.u_gp.cross_amp

        pgs = ls2pr(torch.hstack([gp.lengthscale for gp in self.g_gps]))
        alphas = torch.hstack([gp.alpha for gp in self.g_gps])
        zgs = torch.hstack([gp.inducing_inputs for gp in self.g_gps])
        # this could all be sped up
        bases = [gp.sample_basis() for gp in self.g_gps]
        thetags = torch.hstack([b[0] for b in bases])
        betags = torch.swapaxes(torch.stack([b[1] for b in bases]), 0, 1)
        wgs = torch.swapaxes(torch.stack([b[2] for b in bases]), 0, 1)
        qgs = torch.swapaxes(
            torch.stack(
                [self.g_gps[i].compute_q(bases[i]) for i in range(len(self.g_gps))]
            ),
            0,
            1,
        )

        out = integrals.full_I(
            ts,
            alphas,
            pgs,
            wgs,
            thetags,
            betags,
            zgs,
            qgs,
            pus,
            wus,
            thetaus,
            betaus,
            zus,
            qus,
            ampu,
        )

        layer_out = torch.einsum("oq, bqt -> bto", self.amps, out)

        return layer_out

    def compute_KL(self):
        kl = 0.0
        for g_d in self.g_gps:
            kl += g_d.compute_KL()
        return kl + self.u_gp.compute_KL()

    def plot_features(self, save=None, covariances=True):

        with torch.no_grad():

            fig = plt.figure(
                constrained_layout=False,
                figsize=(2, self.d_in * 2),
            )
            grsp = GridSpec(self.d_in, 1, figure=fig)

            if covariances:
                tgs, g_samps = self.sample_covariance()
            else:
                tgs, g_samps = self.sample_filter()

            for k in range(self.d_in):
                g_ax = fig.add_subplot(grsp[k, 0])
                gm = torch.mean(g_samps[k], axis=0)
                gs = torch.std(g_samps[k], axis=0)
                ts = to_numpy(tgs[k]).flatten()
                g_ax.plot(
                    ts,
                    to_numpy(gm),
                    c=plt.get_cmap("Set2")(k),
                )
                g_ax.fill_between(
                    ts,
                    to_numpy(gm) - to_numpy(gs),
                    to_numpy(gm) + to_numpy(gs),
                    color=plt.get_cmap("Set2")(k),
                    alpha=0.4,
                )
                g_ax.plot(
                    ts,
                    to_numpy(g_samps[k]).T,
                    color=plt.get_cmap("Set2")(k + 1),
                    alpha=0.2,
                )
                if not covariances:
                    g_ax.scatter(
                        to_numpy(self.g_gps[k].inducing_inputs),
                        to_numpy(
                            torch.exp(
                                -self.g_gps[k].alpha
                                * self.g_gps[k].inducing_inputs ** 2
                            )[:, 0]
                            * self.g_gps[k].variational_dist.variational_mean
                        ),
                        color=plt.get_cmap("Set2")(k),
                        alpha=0.7,
                    )
            plt.tight_layout()
            if save is not None:
                plt.savefig(
                    save,
                    dpi=300,
                    bbox_inches="tight",
                )
            else:
                plt.show()
