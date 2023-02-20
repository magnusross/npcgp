import torch

torch.set_default_dtype(torch.float64)
import argparse
import json
import os
import pathlib
from datetime import datetime

import numpy as np

import wandb
from npcgp.data import Datasets
from npcgp.models import NPCGP, FastNPCGP
from npcgp.utils import (
    batch_assess,
    kmeans_initialisations,
    large_scale_kmeans_initialisations,
    to_numpy,
)


def run(args):

    # Seeds & GPU settings
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("SEED =", SEED)

    str_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    save_dir = os.path.join(args.output_dir, "uci", str_time)

    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = (
            True  # Let CUDA optimise strategy, as our inputs are not of variable size
        )
        torch.backends.cudnn.enabled = True
    else:
        device = "cpu"

    print("Device:", device, "\n")

    # Load train & test data
    prop = 0.9
    ds = Datasets(data_path="data/")
    data = ds.all_datasets[args.uci_name].get_data(seed=SEED, prop=prop)
    X_train, y_train, X_test, y_test, y_scale = [
        torch.tensor(data[_], dtype=torch.float64, device=device)
        for _ in ["X", "Y", "Xs", "Ys", "Y_std"]
    ]

    X_valid, y_valid = X_test, y_test

    main_kwargs = {
        "num_basis_functions": 16,
        "mc_samples": 2,
        "init_filter_width": 2.0,
        "num_filter_points": 15,
        "init_noise": 1e-2,
        "init_amp": 1.0,
        "prior_cov_factor_g": 0.8,
        "init_transform_lengthscale": 0.5,
        "jitter": 1e-7,
        "beta": 0.8,
        "prior_mean_factor_g": 0.5,
        "device": device,
    }

    # Initialise input proccess IPs & model
    # num_u_ips = 100
    num_u_ips = 100

    if X_train.shape[0] > 100_000:
        u_ip_inits = large_scale_kmeans_initialisations(num_u_ips, to_numpy(X_train))

    else:
        u_ip_inits = kmeans_initialisations(num_u_ips, to_numpy(X_train))

    u_ip_inits = torch.tensor(u_ip_inits, device=device)

    train_kwargs = {
        "lr": 1e-3,
        "n_iter": args.n_iter,
        "verbosity": args.verbosity,
        "batch_size": args.batch_size,
        "train_time_limit": args.time,
        "fix_g_pars": False,
        "model_filepath": os.path.join(save_dir, "model.torch"),
        "chunks": args.chunks,
    }

    if args.dry_run:
        wandb_mode = "disabled"
    else:
        wandb_mode = "online"

    config = {
        **train_kwargs,
        **main_kwargs,
        "seed": SEED,
        "architecture": "Full",
        "dataset": args.uci_name,
    }

    wandb.init(
        project="npdgp-uci",
        entity="npdgp-paper",
        tags=[args.uci_name],
        config=config,
        mode=wandb_mode,
    )

    npcgp = NPCGP(
        X_train,
        init_u_inducing_inputs=u_ip_inits,
        num_outputs=1,
        **main_kwargs,
    ).to(device)

    wandb.watch(npcgp)

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    npcgp.plot_output_slices(X_train, save=os.path.join(save_dir, "prior_slice.png"))
    npcgp.plot_features(save=os.path.join(save_dir, "prior_features.png"))
    wandb.log(
        {
            "prior_slice": wandb.Image(os.path.join(save_dir, "prior_slice.png")),
        }
    )

    npcgp.train(
        (X_train, y_train),
        (X_valid, y_valid),
        y_scale=y_scale,
        **train_kwargs,
    )
    all_args = {**train_kwargs, **main_kwargs, "num_u_ips": num_u_ips, "seed": SEED}

    with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(all_args, f, ensure_ascii=False, indent=4)

    # Load saved model and compute final test set predictions
    npcgp.load_state_dict(torch.load(os.path.join(save_dir, "model.torch")))

    res_dict = batch_assess(
        npcgp,
        X_test,
        y_test,
        y_scale=y_scale,
        device=device,
        S=500,
    )
    wandb.run.summary["final_rmse"] = res_dict["rmse"]
    wandb.run.summary["final_mnll"] = res_dict["mnll"]

    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    npcgp.plot_output_slices(X_train, save=os.path.join(save_dir, "slice.png"))
    npcgp.plot_features(save=os.path.join(save_dir, "features.png"))
    npcgp.plot_predictions(
        X_test,
        y_test,
        save=os.path.join(save_dir, "preds.png"),
    )

    wandb.log(
        {
            **{
                "slice": wandb.Image(os.path.join(save_dir, "slice.png")),
                "preds": wandb.Image(os.path.join(save_dir, "preds.png")),
            },
            **{f"features": wandb.Image(os.path.join(save_dir, f"features.png"))},
        }
    )

    with open(
        os.path.join(save_dir, "optimised_params.txt"), "w", encoding="utf-8"
    ) as f:
        print("Optimised Model Parameters\n", file=f)
        for param_tensor in npcgp.state_dict():
            print(
                param_tensor,
                "\n",
                npcgp.state_dict()[param_tensor].size(),
                "\n",
                npcgp.state_dict()[param_tensor],
                "\n",
                file=f,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCI experiment.")
    parser.add_argument("--seed", default=1, type=int, help="set random seed")
    parser.add_argument(
        "--uci_name",
        default="power",
        type=str,
        help="choose UCI dataset to run experiment with",
    )
    parser.add_argument(
        "--output_dir",
        default="experiment_outputs",
        type=str,
        help="directory to write outputs to",
    )
    parser.add_argument(
        "--n_iter", default=20000, type=int, help="number of training iterations to run"
    )
    parser.add_argument(
        "--verbosity",
        default=100,
        type=int,
        help="how regularly (in iters) to output metrics during training",
    )
    parser.add_argument(
        "--batch_size", default=1000, type=int, help="set batch size for training"
    )
    parser.add_argument(
        "--time",
        default=50.0,
        type=float,
        help="time limit on training (has priority over n_iters)",
    )
    parser.add_argument(
        "--chunks",
        default=1,
        type=int,
        help="number of chunks to split batches into, if needed due to GPU memory constraints",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="runs experiments without any Weights & Biases logging",
    )
    args = parser.parse_args()
    run(args)
