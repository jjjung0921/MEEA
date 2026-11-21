"""Compare a fixed MLP shortlist, independent SSM, and joint SSM on the policy split."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend((str(ROOT), str(ROOT / "rdchiral")))

import numpy as np
import torch
from torch.nn import functional as F

from joint_tree_policy import TemplateTreeSSM, template_tree_features
from policyNet import RolloutPolicyNet


def load(name):
    with gzip.open(ROOT / "prepare_data" / f"{name}.pkl.gz", "rb") as stream:
        return pickle.load(stream)


def prepare(dataset, proposal, device):
    fp = torch.as_tensor(np.unpackbits(dataset["fp"], axis=1), device=device)
    y = torch.as_tensor(dataset["template"], dtype=torch.long, device=device)
    scores, ids, gold = [], [], []
    with torch.no_grad():
        for x, labels in zip(fp.split(512), y.split(512)):
            output = proposal(x.float())
            top, candidates = output.topk(50)
            scores.append(top)
            ids.append(candidates)
            gold.append(output.gather(1, labels[:, None]))
    return fp, y, torch.cat(scores), torch.cat(ids), torch.cat(gold)


def metrics(scores, candidate_ids, labels):
    positions = scores.argsort(descending=True, dim=-1)
    matches = candidate_ids.gather(1, positions) == labels[:, None]
    return {f"top{k}": matches[:, :k].any(dim=-1).float().mean().item() for k in (1, 10, 50)}


def run(args):
    device = torch.device(args.device)
    rules = load("uspto_template")
    patterns, mask = template_tree_features(rules)
    proposal = RolloutPolicyNet(len(rules)).to(device)
    proposal.load_state_dict(torch.load(ROOT / "results/backbones/mlp.ckpt", map_location=device,
                                        weights_only=True))
    proposal.eval()
    train = prepare(load("policySmilesTrain"), proposal, device)
    valid = prepare(load("policySmilesTest"), proposal, device)
    x, y, prior, ids, gold = train
    vx, vy, vp, vi, _ = valid
    covered = (ids == y[:, None]).any(dim=1)
    print("baseline", metrics(vp, vi, vy), "training shortlist recall", covered.float().mean().item(), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    base = {"baseline": metrics(vp, vi, vy), "seed": args.seed, "train_rows": len(y),
            "val_rows": len(vy), "shortlist": ids.shape[1], "train_shortlist_hits": int(covered.sum())}
    (args.out / "baseline.json").write_text(json.dumps(base, indent=2) + "\n")

    for joint in (False, True):
        torch.manual_seed(args.seed)
        model = TemplateTreeSSM(patterns, mask, joint).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mode = "joint" if joint else "independent"
        best, history = {"top1": -1}, []
        for epoch in range(1, args.epochs + 1):
            model.train()
            for indices in torch.randperm(len(y), device=device).split(args.batch_size):
                candidates = ids[indices].clone()
                priors = prior[indices].clone()
                missing = ~covered[indices]
                candidates[missing, -1] = y[indices][missing]
                priors[missing, -1] = gold[indices][missing, 0]
                label = (candidates == y[indices, None]).long().argmax(1)
                loss = F.cross_entropy(model(x[indices].float(), candidates, priors), label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                output = torch.cat([model(f.float(), c, p) for f, c, p in
                                    zip(vx.split(args.batch_size), vi.split(args.batch_size),
                                        vp.split(args.batch_size))])
                row = {"epoch": epoch, **metrics(output, vi, vy)}
            history.append(row)
            print(mode, row, flush=True)
            if row["top1"] > best["top1"]:
                best = row
                torch.save(model.state_dict(), args.out / f"{mode}.ckpt")
            (args.out / f"{mode}.json").write_text(json.dumps({"best": best, "history": history},
                                                              indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "results/joint_tree")
    run(parser.parse_args())
