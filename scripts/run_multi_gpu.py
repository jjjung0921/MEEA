"""
Simple launcher to run MEEA_A or MEEA_MCTS across GPUs 1, 2, 3 in parallel.

Usage examples:
  python scripts/run_multi_gpu.py --dataset test_dataset/USPTO.pkl --mode a_star --total 150
  python scripts/run_multi_gpu.py --dataset test_dataset/USPTO.pkl --mode mcts --total 150 --cpuct 4.0 --max-rollouts 600
"""

import argparse
import math
import os
import pickle
from multiprocessing import Process

import torch

from MEEA_A import MEEAAStar
from MEEA_MCTS import MEEAMCTS
from MEEA_PC_parallel import prepare_expand, prepare_starting_molecules, prepare_value


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel launcher for MEEA variants over GPUs 1,2,3.")
    parser.add_argument("--dataset", required=True, help="Path to test_dataset .pkl file.")
    parser.add_argument("--mode", choices=["a_star", "mcts"], default="a_star", help="Which variant to run.")
    parser.add_argument("--total", type=int, default=150, help="How many molecules from the dataset to run.")
    parser.add_argument("--policy", default="./saved_model/policy_model.ckpt", help="Policy model checkpoint path.")
    parser.add_argument("--value", default="./saved_model/value_pc.pt", help="Value model checkpoint path.")
    parser.add_argument("--topk", type=int, default=50, help="Top-K reactions from policy model.")
    parser.add_argument("--max-expansions", type=int, default=300, help="A* only: max policy calls/expansions.")
    parser.add_argument("--cpuct", type=float, default=4.0, help="MCTS only: PUCT exploration constant.")
    parser.add_argument("--max-rollouts", type=int, default=600, help="MCTS only: max policy calls/rollouts.")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="MCTS only: Dirichlet alpha at root.")
    parser.add_argument("--dirichlet-frac", type=float, default=0.25, help="MCTS only: Dirichlet weight at root.")
    parser.add_argument(
        "--gpus",
        default="1,2,3",
        help="Comma-separated GPU ids to use (defaults to 1,2,3).",
    )
    return parser.parse_args()


def worker(mode, gpu_id, batch, args):
    # Restrict visible devices so inside the process GPU 0 maps to the assigned device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Load models once per worker to avoid repeatedly loading template rules and weights.
    local_gpu = 0 if gpu_id >= 0 else -1
    device = torch.device("cpu" if local_gpu < 0 else f"cuda:{local_gpu}")
    known_mols = prepare_starting_molecules()
    expand_fn = prepare_expand(args.policy, local_gpu)
    value_model = prepare_value(args.value, local_gpu)

    for smi in batch:
        try:
            if mode == "a_star":
                planner = MEEAAStar(
                    smi,
                    known_mols,
                    value_model,
                    expand_fn,
                    device=device,
                    topk=args.topk,
                    max_expansions=args.max_expansions,
                )
            else:
                planner = MEEAMCTS(
                    smi,
                    known_mols,
                    value_model,
                    expand_fn,
                    device=device,
                    cpuct=args.cpuct,
                    topk=args.topk,
                    max_rollouts=args.max_rollouts,
                    dirichlet_alpha=args.dirichlet_alpha,
                    dirichlet_frac=args.dirichlet_frac,
                )
            success, node, calls = planner.search()
            route, templates = planner.vis_synthetic_path(node)
            depth = node.depth if success and node is not None else -1
            print(
                f"[GPU {gpu_id}] target={smi} success={success} depth={depth} calls={calls} route={route}",
                flush=True,
            )
        except Exception as e:
            print(f"[GPU {gpu_id}] target={smi} failed: {type(e).__name__}: {e}", flush=True)


def main():
    args = parse_args()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]

    mols = pickle.load(open(args.dataset, "rb"))
    if args.total > 0:
        mols = mols[: args.total]

    chunk = math.ceil(len(mols) / len(gpu_ids))
    procs = []
    for i, gpu in enumerate(gpu_ids):
        batch = mols[i * chunk : (i + 1) * chunk]
        if not batch:
            continue
        p = Process(target=worker, args=(args.mode, gpu, batch, args))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
