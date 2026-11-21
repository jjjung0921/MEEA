"""SMILES token encoder comparison: Transformer vs bidirectional diagonal SSM.

Implements experiments/smiles-ssm/plan.md (v1). Both models read the same
tokens, the same vocabulary built from the training split, the same padding
mask, and use the same masked mean pooling and classification head, so only
the token mixing differs. Seed 0, identical batch order, last epoch compared.

  python experiments/smiles-ssm/run.py --verify            # checks only
  python experiments/smiles-ssm/run.py --model transformer  # train one model
  python experiments/smiles-ssm/run.py --compare            # aggregate results
"""
import argparse
import gzip
import hashlib
import json
import os
import pickle
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, "rdchiral"))  # policyNet imports rdchiral.main

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policyNet_variants import (
    PAD,
    SmilesBiSSMPolicyNet,
    SmilesTransformerPolicyNet,
    build_smiles_vocab,
    encode_smiles,
    tokenize_smiles,
)

TRAIN_PKL = os.path.join(_ROOT, "prepare_data", "policySmilesTrain.pkl.gz")
TEST_PKL = os.path.join(_ROOT, "prepare_data", "policySmilesTest.pkl.gz")
TEMPLATE_PKL = os.path.join(_ROOT, "prepare_data", "uspto_template.pkl.gz")
MODELS = ("transformer", "ssm")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj):
    return hashlib.sha256(repr(obj).encode()).hexdigest()


def load_pickle_gz(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(kind, n_rules, vocab_size, max_len):
    if kind == "transformer":
        return SmilesTransformerPolicyNet(
            n_rules=n_rules, vocab_size=vocab_size, max_len=max_len, pool="mean"
        )
    if kind == "ssm":
        return SmilesBiSSMPolicyNet(
            n_rules=n_rules, vocab_size=vocab_size, max_len=max_len
        )
    raise ValueError(kind)


class Split:
    """Token ids of one split, padded per batch to the batch's own longest row."""

    def __init__(self, data, vocab, max_len):
        self.ids = [encode_smiles(s, vocab, max_len) for s in data["smiles"]]
        self.lengths = np.array([len(tokenize_smiles(s)) + 1 for s in data["smiles"]])
        self.y = np.asarray(data["template"], dtype=np.int64)
        self.truncated = int((self.lengths > max_len).sum())

    def __len__(self):
        return len(self.y)

    def batch(self, indices, device):
        seqs = [self.ids[i] for i in indices]
        width = max(len(s) for s in seqs)
        out = np.full((len(seqs), width), PAD, dtype=np.int64)
        for row, s in enumerate(seqs):
            out[row, : len(s)] = s
        return (
            torch.from_numpy(out).to(device),
            torch.from_numpy(self.y[indices]).to(device),
        )


def evaluate(net, split, device, batch_size, want_ranks=False):
    """Validation loss plus the gold rule's rank for every row (1 = best)."""
    net.eval()
    loss_sum, ranks = 0.0, []
    with torch.no_grad():
        for start in range(0, len(split), batch_size):
            indices = np.arange(start, min(start + batch_size, len(split)))
            x, y = split.batch(indices, device)
            logits = net(x).float()
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            gold = logits.gather(1, y[:, None])
            ranks.append((logits > gold).sum(dim=1).add(1).cpu().numpy())
    ranks = np.concatenate(ranks)
    out = {
        "val_loss": loss_sum / len(split),
        "top1": float((ranks <= 1).mean()),
        "top10": float((ranks <= 10).mean()),
        "top50": float((ranks <= 50).mean()),
    }
    return (out, ranks) if want_ranks else (out, None)


def verify(net, split, device, kind):
    """Output shape, padding invariance, finite loss and gradients."""
    indices = np.arange(8)
    x, y = split.batch(indices, device)
    checks = {}

    net.eval()
    with torch.no_grad():
        logits = net(x).float()
        padded = torch.cat(
            [x, torch.full((x.size(0), 7), PAD, dtype=x.dtype, device=device)], dim=1
        )
        logits_padded = net(padded).float()
    checks["logits_shape"] = list(logits.shape)
    checks["logits_shape_ok"] = list(logits.shape) == [len(indices), net.n_rules]
    gap = (logits - logits_padded).abs().max().item()
    checks["padding_max_abs_diff"] = gap
    checks["padding_invariant"] = bool(gap < 1e-4)
    checks["logits_finite"] = bool(torch.isfinite(logits).all())

    net.train()
    net.zero_grad()
    loss = net(x, y)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    checks["loss"] = loss.item()
    checks["loss_finite"] = bool(np.isfinite(loss.item()))
    checks["grad_tensors"] = len(grads)
    checks["grads_finite"] = bool(all(torch.isfinite(g).all() for g in grads))
    checks["grad_max_abs"] = max(g.abs().max().item() for g in grads)
    checks["grad_nonzero"] = bool(any(g.abs().sum().item() > 0 for g in grads))
    net.zero_grad()

    checks["passed"] = all(
        checks[k]
        for k in (
            "logits_shape_ok",
            "padding_invariant",
            "logits_finite",
            "loss_finite",
            "grads_finite",
            "grad_nonzero",
        )
    )
    return {"model": kind, **checks}


def dataset_fingerprint(train_data, test_data, train_split, test_split, vocab):
    return {
        "train_pkl_sha256": sha256_file(TRAIN_PKL),
        "test_pkl_sha256": sha256_file(TEST_PKL),
        "template_pkl_sha256": sha256_file(TEMPLATE_PKL),
        "n_train": len(train_split),
        "n_val": len(test_split),
        "train_label_sha256": sha256_obj(train_data["template"].tolist()),
        "val_label_sha256": sha256_obj(test_data["template"].tolist()),
        "train_smiles_order_sha256": sha256_obj(list(train_data["smiles"])),
        "val_smiles_order_sha256": sha256_obj(list(test_data["smiles"])),
        "train_rule_ids": sorted(set(int(v) for v in train_data["template"])).__len__(),
        "val_rule_ids": sorted(set(int(v) for v in test_data["template"])).__len__(),
        "val_rules_unseen_in_train": int(
            len(set(test_data["template"].tolist()) - set(train_data["template"].tolist()))
        ),
        "val_rows_with_unseen_rule": int(
            np.isin(
                test_data["template"], list(set(test_data["template"].tolist()) - set(train_data["template"].tolist()))
            ).sum()
        ),
        "vocab_size": len(vocab),
        "vocab_sha256": sha256_obj(sorted(vocab.items())),
        "train_truncated_rows": train_split.truncated,
        "val_truncated_rows": test_split.truncated,
    }


def train_one(kind, args, train_split, test_split, n_rules, vocab, device, out_dir):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net = build_model(kind, n_rules, len(vocab), args.max_len).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6)
    # Seeded separately from model init so both models see the same batch order.
    order = torch.Generator().manual_seed(args.seed)

    log = {
        "model": kind,
        "device": str(device),
        "params": n_params,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_len": args.max_len,
        "seed": args.seed,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau(val_loss)",
        "grad_clip": args.grad_clip,
        "history": [],
    }
    print(json.dumps({k: v for k, v in log.items() if k != "history"}), flush=True)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(len(train_split), generator=order).numpy()
        if epoch == 1:
            log["first_epoch_order_sha256"] = sha256_obj(perm[:1000].tolist())
        losses = []
        for start in range(0, len(perm), args.batch_size):
            indices = perm[start : start + args.batch_size]
            x, y = train_split.batch(indices, device)
            optimizer.zero_grad()
            loss = net(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            losses.append(loss.item())
            if args.max_batches and len(losses) >= args.max_batches:
                break

        want_ranks = epoch == args.epochs
        metrics, ranks = evaluate(net, test_split, device, args.batch_size, want_ranks)
        scheduler.step(metrics["val_loss"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(time.time() - t0, 1),
        }
        log["history"].append(row)
        print(json.dumps(row), flush=True)

        log["last_epoch"] = row
        log["train_time_s"] = round(time.time() - t0, 1)
        with open(os.path.join(out_dir, f"{kind}.json"), "w") as f:
            json.dump(log, f, indent=2)
        torch.save(net.state_dict(), os.path.join(out_dir, f"{kind}.ckpt"))
        if ranks is not None:
            np.save(os.path.join(out_dir, f"{kind}_val_gold_rank.npy"), ranks)
    return log


def compare(out_dir, test_data):
    logs, ranks = {}, {}
    for kind in MODELS:
        with open(os.path.join(out_dir, f"{kind}.json")) as f:
            logs[kind] = json.load(f)
        ranks[kind] = np.load(os.path.join(out_dir, f"{kind}_val_gold_rank.npy"))

    labels = np.asarray(test_data["template"], dtype=np.int64)
    summary = {"n_val": int(len(labels)), "models": {}, "paired": {}}
    for kind in MODELS:
        log = logs[kind]
        summary["models"][kind] = {
            "params": log["params"],
            "train_time_s": log["train_time_s"],
            "epochs_run": log["last_epoch"]["epoch"],
            "last_epoch": log["last_epoch"],
            "mean_gold_rank": float(ranks[kind].mean()),
            "median_gold_rank": float(np.median(ranks[kind])),
        }

    a, b = (ranks[k] for k in MODELS)
    for k in (1, 50):
        hit_a, hit_b = a <= k, b <= k
        summary["paired"][f"top{k}"] = {
            MODELS[0]: float(hit_a.mean()),
            MODELS[1]: float(hit_b.mean()),
            "difference": float(hit_b.mean() - hit_a.mean()),
            "both": int((hit_a & hit_b).sum()),
            "neither": int((~hit_a & ~hit_b).sum()),
            f"{MODELS[0]}_only": int((hit_a & ~hit_b).sum()),
            f"{MODELS[1]}_only": int((~hit_a & hit_b).sum()),
            "exactly_one": int((hit_a ^ hit_b).sum()),
        }

    rows = [
        {"index": i, "rule_id": int(labels[i]),
         f"{MODELS[0]}_rank": int(a[i]), f"{MODELS[1]}_rank": int(b[i])}
        for i in range(len(labels))
    ]
    with open(os.path.join(out_dir, "val_gold_ranks.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-len", type=int, default=160)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-batches", type=int, default=0, help="smoke test: cap batches per epoch")
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_data = load_pickle_gz(TRAIN_PKL)
    test_data = load_pickle_gz(TEST_PKL)
    n_rules = len(load_pickle_gz(TEMPLATE_PKL))

    if args.compare:
        compare(args.out_dir, test_data)
        return

    vocab = build_smiles_vocab(train_data["smiles"])
    train_split = Split(train_data, vocab, args.max_len)
    test_split = Split(test_data, vocab, args.max_len)
    device = pick_device(args.device)

    fingerprint = dataset_fingerprint(train_data, test_data, train_split, test_split, vocab)
    fingerprint["n_rules"] = n_rules
    fingerprint["device"] = str(device)
    with open(os.path.join(args.out_dir, "dataset.json"), "w") as f:
        json.dump(fingerprint, f, indent=2)
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    print(json.dumps(fingerprint, indent=2), flush=True)

    if args.verify:
        report = []
        for kind in MODELS:
            torch.manual_seed(args.seed)
            net = build_model(kind, n_rules, len(vocab), args.max_len).to(device)
            checks = verify(net, train_split, device, kind)
            checks["params"] = sum(p.numel() for p in net.parameters())
            report.append(checks)
            print(json.dumps(checks), flush=True)
        with open(os.path.join(args.out_dir, "verification.json"), "w") as f:
            json.dump(report, f, indent=2)
        if not all(c["passed"] for c in report):
            raise SystemExit("verification failed; holding the score comparison")
        return

    if not args.model:
        raise SystemExit("pass --model, --verify or --compare")
    train_one(args.model, args, train_split, test_split, n_rules, vocab, device, args.out_dir)


if __name__ == "__main__":
    main()
