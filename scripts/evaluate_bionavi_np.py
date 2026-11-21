"""Evaluate expansion policies with MEEA*-PC on the 388-target BioNavi-NP set."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import io
import json
import multiprocessing
from pathlib import Path
import pickle
import statistics
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "rdchiral"),
                str(ROOT / "data/raw/BioNavi-NP/multistep/onmt")]

import torch

from MEEA_PC_NPs_parallel import MCTS_A, TimeoutException, time_limit
from valueEnsemble import ValueEnsemble

ARCHIVE = ROOT / "data/raw/meea_author_release.zip"
ASSETS = ROOT / "data/raw/bionavi_assets/MEEA_model"
DEVICE = "cuda:0"
MAX_CALLS = 100
MAX_EXPANSIONS = 50


class BioNaviPolicy:
    def __init__(self):
        from onmt.bin.parser import Seq2seqArgs
        from onmt.bin.translate import run
        from onmt.translate.translator import build_translator

        self.translate = run
        opt = Seq2seqArgs()
        opt.models = [str(ASSETS / "checkpoints" / f"model_step_{n}.pt")
                      for n in (30000, 50000, 80000, 100000)]
        opt.gpu = 0
        opt.beam_size = opt.n_best = 20
        opt.topk = 50
        opt.max_length = 200
        opt.tokenizer = "char"
        self.output = io.StringIO()
        self.translator = build_translator(opt, report_score=False, out_file=self.output)
        self.opt = opt

    def run(self, smiles):
        self.output.seek(0)
        self.output.truncate()
        predictions = self.translate(self.translator, self.opt, smiles)
        if not predictions["scores"]:
            return None
        return {**predictions, "template": [None] * len(predictions["scores"])}


class TemplatePolicy:
    def __init__(self, name):
        from scripts.benchmark_policy_search import SearchPolicy

        with gzip.open(ROOT / "prepare_data/uspto_template.pkl.gz", "rb") as stream:
            rules = pickle.load(stream)
        self.policy = SearchPolicy(name, rules, device=DEVICE)

    def run(self, smiles):
        return self.policy.expand(smiles, topk=50)


def initialize(policy_name):
    global policy, value, stock
    torch.set_num_threads(1)
    policy = BioNaviPolicy() if policy_name == "bionavi_np" else TemplatePolicy(policy_name)
    with ZipFile(ARCHIVE) as archive:
        stock = {line.strip().split("\t")[-1] for line in archive.read(
            "MEEA_model/prepare_data/building_blocks.txt").decode().splitlines() if line.strip()}
        with archive.open("MEEA_model/saved_model/value_pc.pt") as stream:
            state = torch.load(stream, map_location="cpu", weights_only=True)
    value = ValueEnsemble(2048, 128, 0.1)
    value.load_state_dict(state)
    value.to(DEVICE).eval()


@torch.inference_mode()
def search(job):
    index, target = job
    start = time.monotonic()
    player = None
    try:
        with time_limit(600):
            player = MCTS_A(target, stock, value, policy, DEVICE, 100, 4.0)
            player.last_logged_iteration = MAX_CALLS
            success, end, _ = player.search(MAX_CALLS, max_expansions=MAX_EXPANSIONS)
            route, _ = player.vis_synthetic_path(end)
    except TimeoutException:
        status, route = "timeout", []
    else:
        status = "solved" if success else "unsolved"
    return {"index": index, "status": status, "calls": player.iterations if player else 0,
            "expansions": getattr(player, "expansions", 0),
            "steps": sum(bool(r) for r in route) if route else None,
            "seconds": round(time.monotonic() - start, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("bionavi_np", "smiles_transformer", "smiles_bond_changes"),
                        default="bionavi_np")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    baseline = ROOT / "results/bionavi_np"
    out = args.out or (baseline if args.policy == "bionavi_np" else baseline / args.policy)
    out.mkdir(parents=True, exist_ok=True)
    records = out / "targets.jsonl"
    manifest = out / "policy.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != args.policy:
            parser.error("output directory contains results from another policy")
    elif records.exists() and args.policy != "bionavi_np":
        parser.error("existing results have no policy manifest")
    manifest.write_text(json.dumps(args.policy) + "\n")
    with (ROOT / "test_dataset/natural.pkl").open("rb") as stream:
        targets = pickle.load(stream)
    rows = list(map(json.loads, records.read_text().splitlines())) if records.exists() else []
    done = {row["index"] for row in rows}
    jobs = [(i, target) for i, target in enumerate(targets) if i not in done]
    if jobs:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize, initargs=(args.policy,)) as pool, records.open("a", buffering=1) as stream:
            for future in as_completed([pool.submit(search, job) for job in jobs]):
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                if len(rows) % 8 == 0 or len(rows) == len(targets):
                    print(f"{len(rows)}/{len(targets)} complete", flush=True)
    solved = [r for r in rows if r["status"] == "solved"]
    summary = {"targets": len(rows), "solved": len(solved), "success_rate": len(solved) / len(rows),
               "timeouts": sum(r["status"] == "timeout" for r in rows),
               "mean_steps": statistics.mean(r["steps"] for r in solved) if solved else None,
               "max_policy_calls": MAX_CALLS, "max_expansions": MAX_EXPANSIONS,
               "simulations": 100, "cpuct": 4.0, "beam_size": 20}
    summary["policy"] = args.policy
    if args.policy != "bionavi_np":
        del summary["beam_size"]
        summary["max_candidate_templates"] = 50
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
