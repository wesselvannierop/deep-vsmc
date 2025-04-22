import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DPF")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="/app/dpf/configs/dpf.yaml",
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


import dpf.runnable  # isort: skip

args = parse_args()
dpf.runnable.runnable(args.backend, args.device)

from dpf.learned_pf import dpf_run
from experiments import debugging, setup_experiment

if __name__ == "__main__":

    config, run = setup_experiment(args.config, debug_run=args.debug_run)
    if (args.debug_run or debugging) and config.data.get("limit_n_samples") is None:
        # to also test checkpointing & validation
        config.data.limit_n_samples = int(config.data.batch_size * 1.5)

    if config.experiment == "lorenz":
        from dpf.data.lorenz import fast_debug_config

        fast_debug_config(config, debugging=debugging)

    pf = dpf_run(config)
