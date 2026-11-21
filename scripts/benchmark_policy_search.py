"""Compare trained policies on matched MEEA*-PC organic benchmarks."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import gzip
import json
import multiprocessing
from pathlib import Path
import pickle
import statistics
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend((str(ROOT), str(ROOT / "rdchiral")))

import torch
from rdkit import Chem
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from rdchiral.main import rdchiralRun

import MEEA_PC_parallel as planner
from MEEA_PC_parallel import MCTS_A, TimeoutException, prepare_starting_molecules, time_limit
from joint_tree_policy import ShortlistPolicy, TemplateTreeSSM, template_tree_features
from policyNet import RolloutPolicyNet, merge, preprocess
from policyNet_variants import SmilesTransformerPolicyNet, TransformerPolicyNet, encode_smiles
from valueEnsemble import ValueEnsemble


POLICIES = {
    "mlp": ("backbones", "mlp"),
    "fp_transformer": ("backbones", "fp_transformer"),
    "smiles_transformer": ("backbones", "smiles_transformer"),
    "smiles_bond_changes": ("bond_changes", "smiles_transformer"),
}


ORGANIC_DATASETS = ("USPTO", "logS", "logP", "Toxicity_LD50", "Ames", "BBBP",
                    "DPP4", "bace", "SVS", "ClinTox")
TREE_POLICIES = {"independent_tree_ssm": "independent", "joint_tree_ssm": "joint"}


class SearchPolicy:
    def __init__(self, name, rules, tree_features=None, device="cpu"):
        experiment, model = POLICIES["mlp" if name in TREE_POLICIES else name]
        directory = ROOT / "results" / experiment
        self.rules = rules
        self.device = device
        self.vocab = None
        self.max_len = 160
        if model == "smiles_transformer":
            self.vocab = json.loads((directory / f"{model}_vocab.json").read_text())
            self.net = SmilesTransformerPolicyNet(len(rules), len(self.vocab), max_len=self.max_len)
        elif model == "fp_transformer":
            self.net = TransformerPolicyNet(len(rules))
        else:
            self.net = RolloutPolicyNet(len(rules))
        self.net.load_state_dict(torch.load(directory / f"{model}.ckpt", map_location="cpu", weights_only=True))
        if name in TREE_POLICIES:
            mode = TREE_POLICIES[name]
            features, masks = tree_features
            ranker = TemplateTreeSSM(features, masks, joint=(mode == "joint"))
            ranker.load_state_dict(torch.load(ROOT / "results/joint_tree" / f"{mode}.ckpt",
                                               map_location="cpu", weights_only=True))
            self.net = ShortlistPolicy(self.net, ranker)
        self.net.to(device).eval()
        self.compile_rule = lru_cache(maxsize=4096)(rdchiralReaction)
        self.run = lru_cache(maxsize=8192)(self.expand)

    @torch.inference_mode()
    def expand(self, smiles, topk):
        if self.vocab is None:
            x = torch.as_tensor(preprocess(smiles, 2048), dtype=torch.float32)[None, :]
        else:
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            x = torch.tensor([encode_smiles(canonical, self.vocab, self.max_len)])
        scores, indices = self.net(x.to(self.device)).softmax(dim=1)[0].topk(topk)
        reactants = rdchiralReactants(smiles)
        candidates = defaultdict(list)
        for score, index in zip(scores.tolist(), indices.tolist()):
            rule = self.rules[index]
            try:
                outcomes = rdchiralRun(self.compile_rule(rule), reactants)
            except (ValueError, RuntimeError, KeyError, IndexError):
                continue
            for outcome in sorted(outcomes):
                key = ".".join(sorted(outcome.split(".")))
                candidates[key].append((score / len(outcomes), rule))
        if not candidates:
            return None
        precursors, weights, rules = merge(candidates)
        total = sum(weights)
        return {"reactants": precursors, "scores": [w / total for w in weights], "template": rules}


def init_worker(archive, rules, tree_features):
    global value_model, policies
    torch.set_num_threads(1)
    planner.smiles_to_fp = lru_cache(maxsize=8192)(planner.smiles_to_fp)
    with ZipFile(archive) as bundle, bundle.open("MEEA_model/saved_model/value_pc.pt") as stream:
        weights = torch.load(stream, map_location="cpu", weights_only=True)
    value_model = ValueEnsemble(2048, 128, 0.1)
    value_model.load_state_dict(weights)
    device = settings.get("device", "cpu")
    value_model.to(device).eval()
    policies = {name: SearchPolicy(name, rules, tree_features, device) for name in selected_policies}


@torch.inference_mode()
def search(job):
    name, index, smiles = job
    if settings.get("paper"):
        policies[name].run.cache_clear()
    start = time.monotonic()
    player = None
    try:
        with time_limit(settings["timeout"]):
            player = MCTS_A(smiles, stock, value_model, policies[name], settings.get("device", "cpu"),
                            settings["simulations"], settings["cpuct"],
                            batch_values=batch_values)
            # This runner records progress per target in the parent.
            player.last_logged_iteration = settings["iterations"]
            success, end, _ = player.search(settings["iterations"])
            route, _ = player.vis_synthetic_path(end)
    except TimeoutException:
        success, end, route = False, None, []
        status = "timeout"
    else:
        status = "solved" if success else "unsolved"
    return {"policy": name, "index": index, "target": smiles, "status": status,
            "iterations": player.iterations if player else 0,
            "steps": end.depth if end else None, "route": [r for r in route if r],
            "seconds": round(time.monotonic() - start, 3),
            "batch_values": batch_values}


def main():
    global stock, settings, selected_policies, batch_values
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", nargs="+", choices=(*POLICIES, *TREE_POLICIES),
                        default=list(POLICIES))
    parser.add_argument("--dataset", choices=ORGANIC_DATASETS, default="USPTO")
    parser.add_argument("--paper", action="store_true",
                        help="500 calls, 600 seconds, 100 simulations, c_puct=4, full target set")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-values", action="store_true",
                        help="batch sibling state cost estimates within each expansion")
    parser.add_argument("--device", default="cpu", help="PyTorch inference device, e.g. cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--cpuct", type=float, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    selected_policies = args.policies
    batch_values = args.batch_values
    settings = {k: getattr(args, k) for k in ("iterations", "simulations", "cpuct", "timeout")}
    if args.paper:
        settings = dict(iterations=500, simulations=100, cpuct=4, timeout=600, paper=True)
        if args.limit is not None:
            parser.error("--paper uses the entire dataset")
    if args.device != "cpu":
        settings["device"] = str(torch.device(args.device))
    with (ROOT / "test_dataset" / f"{args.dataset}.pkl").open("rb") as stream:
        targets = pickle.load(stream)[:args.limit]
    with gzip.open(ROOT / "prepare_data/uspto_template.pkl.gz", "rb") as stream:
        rules = pickle.load(stream)
    args.out.mkdir(parents=True, exist_ok=True)
    progress = args.out / "targets.jsonl"
    config = {"policies": selected_policies, "targets": len(targets), **settings}
    if args.dataset != "USPTO" or args.paper:
        config.update(dataset=args.dataset, topk=50)
    config_file = args.out / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError("Use a different output folder for different search settings")
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    rows = [json.loads(line) for line in progress.read_text().splitlines()] if progress.exists() else []
    done = {(row["policy"], row["index"]) for row in rows}
    jobs = [(name, i, mol) for i, mol in enumerate(targets) for name in selected_policies
            if (name, i) not in done]
    if jobs:
        stock = prepare_starting_molecules()
        print(f"Loaded {len(stock):,} stock molecules. {len(jobs)} searches pending.", flush=True)
        archive = ROOT / "data/raw/meea_author_release.zip"
        tree_features = template_tree_features(rules) if TREE_POLICIES.keys() & set(selected_policies) else None
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork"),
                                 initializer=init_worker, initargs=(archive, rules, tree_features)) as executor:
            with progress.open("a", buffering=1) as stream:
                futures = [executor.submit(search, job) for job in jobs]
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    if len(rows) % 16 == 0 or len(rows) == len(targets) * len(selected_policies):
                        print(f"{len(rows)}/{len(targets) * len(selected_policies)} complete", flush=True)
    summary = []
    for name in selected_policies:
        policy_rows = [row for row in rows if row["policy"] == name]
        solved = [row for row in policy_rows if row["status"] == "solved"]
        summary.append({"policy": name, "targets": len(policy_rows), "solved": len(solved),
                        "timeouts": sum(row["status"] == "timeout" for row in policy_rows),
                        "success_rate": len(solved) / len(policy_rows),
                        "multistep_solved": sum(row["steps"] >= 2 for row in solved),
                        "mean_steps_on_success": statistics.mean(row["steps"] for row in solved) if solved else None})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
