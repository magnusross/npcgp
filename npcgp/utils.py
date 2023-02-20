import torch
import numpy as np
from math import pi
from sklearn.cluster import KMeans, MiniBatchKMeans


def to_numpy(tensor):
    """Convert GPU tensor to NumPy array."""
    return tensor.detach().cpu().numpy()


def ls2pr(ls):
    """Convert lengthscale to precision."""
    return 1 / (2 * ls**2)


def pr2ls(pr):
    """Convert precision to lengthscale."""
    return (1 / (2 * pr)) ** 0.5


def batch_assess(
    model, X, Y, y_scale=None, device="cpu", batch_size=1000, S=100, task="regression"
):
    """
    Evaluates batches of inputs and computes metrics accordingly, adapted from:
    https://github.com/ICL-SML/Doubly-Stochastic-DGP/blob/master/demos/demo_regression_UCI.ipynb
    """
    if y_scale is None:
        y_scale = 1.0

    n_batches = max(int(X.shape[0] / batch_size), 1)

    if task == "regression":
        all_metrics = {"mnll": [], "mse": []}
    elif task == "classification":
        all_metrics = {"mnll": [], "acc": [], "auc": []}

    for X_batch, Y_batch in zip(
        np.array_split(X, n_batches), np.array_split(Y, n_batches)
    ):
        _, _, metrics = model.predict(
            X_batch.to(device),
            y=Y_batch.to(device),
            y_scale=y_scale,
            S=S,
        )
        # [all_metrics[key].append(val) for (key, val) in metrics]

        if task == "regression":
            all_metrics["mnll"].append(metrics["mnll"])
            all_metrics["mse"].append(metrics["mse"])
        elif task == "classification":
            all_metrics["mnll"].append(metrics["mnll"])
            all_metrics["acc"].append(metrics["acc"])
            all_metrics["auc"].append(metrics["auc"])

    if task == "regression":
        all_metrics["mse"] = np.mean(np.array(all_metrics["mse"]))
        all_metrics["rmse"] = np.sqrt(all_metrics["mse"])
        all_metrics["mnll"] = np.mean(np.array(all_metrics["mnll"]))
    elif task == "classification":
        all_metrics["acc"] = np.mean(np.array(all_metrics["acc"]))
        all_metrics["mnll"] = np.mean(np.array(all_metrics["mnll"]))
        all_metrics["auc"] = np.mean(np.array(all_metrics["auc"]))

    return all_metrics


def kmeans_initialisations(num_inducing_points, X):
    """KMeans for initialising inducing inputs."""
    kmeans = KMeans(n_clusters=num_inducing_points).fit(X)
    return kmeans.cluster_centers_


def large_scale_kmeans_initialisations(num_inducing_points, X):
    """KMeans for initialising inducing inputs based on a large amount of input data."""
    kmeans = MiniBatchKMeans(n_clusters=num_inducing_points, verbose=True).fit(X)
    return kmeans.cluster_centers_


def approx_prior_ls(alpha, pg, pu):
    """Generate approximation of the prior lengthscale based on model parameters."""
    pr = ((alpha + 2 * pg) * pu) / (g_gp.alpha + 2 * (pg + pu))
    return pr2ls(pr)


def approx_prior_norm(alpha, pg, pu):
    return pi / (alpha * (alpha + 2 * (pg + pu))) ** 0.5


def double_integral(xmin, xmax, ymin, ymax, nx, ny, A):
    """
    Compute the double integral required to plot the form of the covariance. See
    https://stackoverflow.com/questions/20668689/integrating-2d-samples-on-a-rectangular-grid-using-scipy
    """

    dS = ((xmax - xmin) / (nx - 1)) * ((ymax - ymin) / (ny - 1))

    A_Internal = A[:, 1:-1, 1:-1]

    # sides: up, down, left, right
    (A_u, A_d, A_l, A_r) = (
        A[:, 0, 1:-1],
        A[:, -1, 1:-1],
        A[:, 1:-1, 0],
        A[:, 1:-1, -1],
    )

    # corners
    (A_ul, A_ur, A_dl, A_dr) = (
        A[:, 0, 0],
        A[:, 0, -1],
        A[:, -1, 0],
        A[:, -1, -1],
    )  # dt size vector

    return dS * (
        torch.sum(A_Internal, axis=(1, 2))
        + 0.5
        * (
            torch.sum(A_u, axis=1)
            + torch.sum(A_d, axis=1)
            + torch.sum(A_l, axis=1)
            + torch.sum(A_r, axis=1)
        )
        + 0.25 * (A_ul + A_ur + A_dl + A_dr)
    )
