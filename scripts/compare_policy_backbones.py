"""
Train one policy backbone on the SMILES-aligned subset and log per-epoch metrics.

All three backbones see exactly the same reactions, labels and split
(prepare_data/policySmiles{Train,Test}.pkl.gz), so only the input
representation and the architecture differ:

  mlp                 fingerprint bits  -> MLP (the paper's RolloutPolicyNet)
  fp_transformer      fingerprint bits  -> patchified Transformer (slide 14-16)
  smiles_transformer  SMILES tokens     -> Transformer encoder (slide 15 idea)

Usage:
  python scripts/compare_policy_backbones.py --model smiles_transformer --epochs 30
"""
import argparse
import gzip
import json
import os
import pickle
import sys
import time
from functools import partial

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, "rdchiral"))  # policyNet imports rdchiral.main

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from policyNet import RolloutPolicyNet, top_k_acc
from policyNet_variants import (
    PAD,
    SmilesTransformerPolicyNet,
    TransformerPolicyNet,
    build_smiles_vocab,
    encode_smiles,
)


class FpDataset(Dataset):
    def __init__(self, data):
        self.x, self.y = data["fp"], data["template"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return np.unpackbits(self.x[i]).astype(np.float32), self.y[i]


class SmilesDataset(Dataset):
    def __init__(self, data, vocab, max_len):
        self.ids = [encode_smiles(s, vocab, max_len) for s in data["smiles"]]
        self.y = data["template"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.ids[i], self.y[i]


def collate_fp(batch):
    x, y = zip(*batch)
    return torch.tensor(np.array(x)), torch.tensor(np.array(y), dtype=torch.int64)


def collate_smiles(batch):
    seqs, y = zip(*batch)
    n = max(len(s) for s in seqs)
    out = np.full((len(seqs), n), PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return torch.tensor(out), torch.tensor(np.array(y), dtype=torch.int64)


class BondChangeDataset(Dataset):
    def __init__(self, policy_data, targets):
        self.policy_data = policy_data
        self.targets = targets

    def __len__(self):
        return len(self.policy_data)

    def __getitem__(self, index):
        return self.policy_data[index], self.targets[index]


def collate_bond_changes(batch, collate):
    policy_rows, changes = zip(*batch)
    return (*collate(policy_rows), torch.tensor(np.asarray(changes), dtype=torch.float32))


def metrics_by_frequency(net, loader, device, amp, frequency):
    net.eval()
    counts = np.zeros((4, 4), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            with torch.cuda.amp.autocast(enabled=amp):
                logits = net(x.to(device))
            ranks = logits.float().topk(50).indices.cpu() == y[:, None]
            groups = np.digitize(frequency[y.numpy()], [1, 2, 6])
            np.add.at(counts[:, 0], groups, 1)
            for index, k in enumerate((1, 10, 50), start=1):
                np.add.at(counts[:, index], groups, ranks[:, :k].any(1).numpy())
    return {
        name: {"n": int(n), **{
            f"top{k}": int(hits) / int(n) if n else None
            for k, hits in zip((1, 10, 50), row)
        }}
        for name, (n, *row) in zip(("0", "1", "2-5", "6+"), counts)
    }


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(net, loader, device, amp=False):
    net.eval()
    c1 = c10 = c50 = n = 0
    loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = net(xb)
            logits = logits.float()
            loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
            for k, acc in ((1, "c1"), (10, "c10"), (50, "c50")):
                correct, num = top_k_acc(logits, yb, k=k)
                if k == 1:
                    c1 += correct
                elif k == 10:
                    c10 += correct
                else:
                    c50 += correct
            n += yb.size(0)
    return c1 / n, c10 / n, c50 / n, loss_sum / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["mlp", "fp_transformer", "smiles_transformer"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-len", type=int, default=160)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true",
                   help="mixed precision (CUDA only; use on the RTX desktop)")
    p.add_argument("--num-workers", type=int, default=-1,
                   help="DataLoader workers; -1 = 4 on CUDA, 0 otherwise")
    p.add_argument("--out-dir", default="results/backbones")
    p.add_argument("--bond-targets", help="training-aligned bond target NPZ")
    p.add_argument("--aux-weight", type=float, default=0.25)
    p.add_argument("--eval-checkpoint", help="report validation frequency buckets and exit")
    args = p.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    with gzip.open("prepare_data/policySmilesTrain.pkl.gz", "rb") as f:
        train_data = pickle.load(f)
    with gzip.open("prepare_data/policySmilesTest.pkl.gz", "rb") as f:
        test_data = pickle.load(f)
    with gzip.open("prepare_data/uspto_template.pkl.gz", "rb") as f:
        n_rules = len(pickle.load(f))

    device = pick_device(args.device)
    vocab = None

    if args.model == "smiles_transformer":
        vocab = build_smiles_vocab(train_data["smiles"])
        train_ds = SmilesDataset(train_data, vocab, args.max_len)
        test_ds = SmilesDataset(test_data, vocab, args.max_len)
        collate = collate_smiles
        net = SmilesTransformerPolicyNet(
            n_rules=n_rules, vocab_size=len(vocab), max_len=args.max_len
        )
    else:
        train_ds, test_ds = FpDataset(train_data), FpDataset(test_data)
        collate = collate_fp
        net = (RolloutPolicyNet(n_rules=n_rules)
               if args.model == "mlp"
               else TransformerPolicyNet(n_rules=n_rules))

    cuda = device.type == "cuda"
    workers = (4 if cuda else 0) if args.num_workers < 0 else args.num_workers
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=workers,
                             pin_memory=cuda, persistent_workers=workers > 0)

    net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    amp = args.amp and cuda
    frequency = np.bincount(train_data["template"], minlength=n_rules)
    if args.eval_checkpoint:
        net.load_state_dict(torch.load(args.eval_checkpoint, map_location=device, weights_only=True))
        print(json.dumps(metrics_by_frequency(net, test_loader, device, amp, frequency)))
        return

    auxiliary = None
    train_collate = collate
    parameters = list(net.parameters())
    if args.bond_targets:
        with np.load(args.bond_targets) as data:
            targets = data["targets"]
            if not np.array_equal(data["labels"], train_data["template"]):
                raise ValueError("Bond targets do not match the policy training labels")
        train_ds = BondChangeDataset(train_ds, targets)
        train_collate = partial(collate_bond_changes, collate=collate)
        width = net.fc3.in_features if args.model == "mlp" else net.head[-1].in_features
        with torch.random.fork_rng(devices=[]):
            auxiliary = nn.Linear(width, targets.shape[1]).to(device)
        parameters.extend(auxiliary.parameters())
        # Each positive edit contributes equally, independent of vocabulary size.
        positive_scale = targets.shape[1] / targets.sum(axis=1).mean()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_collate, num_workers=workers,
                              pin_memory=cuda, persistent_workers=workers > 0)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    os.makedirs(args.out_dir, exist_ok=True)
    log = {"model": args.model, "device": str(device), "params": n_params,
           "amp": amp, "gpu": torch.cuda.get_device_name(device) if cuda else None,
           "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
           "n_train": len(train_ds), "n_test": len(test_ds), "n_rules": n_rules,
           "vocab_size": len(vocab) if vocab else None, "history": []}
    if auxiliary is not None:
        log.update(aux_weight=args.aux_weight, aux_targets=targets.shape[1])
    print(json.dumps({k: v for k, v in log.items() if k != "history"}))

    best = {"top1": -1.0}
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for xb, yb, *extra in train_loader:
            xb = xb.to(device, non_blocking=cuda)
            yb = yb.to(device, non_blocking=cuda)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp):
                if auxiliary is None:
                    loss = net(xb, yb).mean()
                else:
                    representation = net.encode(xb)
                    logits = net.fc3(representation) if args.model == "mlp" else net.head(representation)
                    loss = F.cross_entropy(logits, yb)
                    edits = extra[0].to(device, non_blocking=cuda)
                    aux_logits = auxiliary(representation)
                    weights = 1 + edits * (positive_scale - 1)
                    aux_loss = F.binary_cross_entropy_with_logits(aux_logits, edits, weight=weights)
                    loss = loss + args.aux_weight * aux_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, max_norm=5)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        top1, top10, top50, val_loss = evaluate(net, test_loader, device, amp)
        if cuda:
            torch.cuda.synchronize()
        scheduler.step(val_loss)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "val_loss": val_loss, "top1": top1, "top10": top10,
               "top50": top50, "elapsed_s": time.time() - t0}
        log["history"].append(row)
        print(json.dumps(row), flush=True)

        if top1 > best["top1"]:
            best = row
            torch.save(net.state_dict(), f"{args.out_dir}/{args.model}.ckpt")
            if vocab is not None:
                with open(f"{args.out_dir}/{args.model}_vocab.json", "w") as f:
                    json.dump(vocab, f)

        log["best"] = best
        with open(f"{args.out_dir}/{args.model}.json", "w") as f:
            json.dump(log, f, indent=2)

    if auxiliary is not None:
        net.load_state_dict(torch.load(f"{args.out_dir}/{args.model}.ckpt", map_location=device,
                                       weights_only=True))
        log["best_by_frequency"] = metrics_by_frequency(net, test_loader, device, amp, frequency)
        with open(f"{args.out_dir}/{args.model}.json", "w") as f:
            json.dump(log, f, indent=2)
    print("BEST", json.dumps(best))


if __name__ == "__main__":
    main()
