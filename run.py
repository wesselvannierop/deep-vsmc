import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DPF")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/train_lorenz_awgn.yaml",
        help="Path to the config",
    )
    parser.add_argument(
        "--debug_run",
        action="store_true",
        help="Run normally, but use wandb offline and add debug flag to the save path",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="tensorflow",
        help="Backend to use (tensorflow or jax)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto:1",
        help="Device to use (auto:1, auto:2, cpu, cuda:0, ...), lorenz with elbo on tf needs auto:2",
    )
    return parser.parse_args()


import vsmc.runnable  # isort: skip

args = parse_args()
vsmc.runnable.runnable(args.backend, args.device)

from vsmc.experiments import debugging, setup_experiment
from vsmc.learned_pf import dpf_run

if __name__ == "__main__":

    config, run = setup_experiment(args.config, debug_run=args.debug_run)

    pf = dpf_run(config)
