"""
One command to reproduce the fingerprint-vs-SMILES policy comparison on a GPU box.

Builds the SMILES dataset if it is missing, trains every backbone with the same
data/split/epoch budget, then writes a comparison table.

  python scripts/run_backbone_benchmark.py --epochs 30 --batch-size 512 --amp

Add --models smiles_transformer to train only one of them.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["mlp", "fp_transformer", "smiles_transformer"]
LABEL = {
    "mlp": "fingerprint -> MLP (baseline)",
    "fp_transformer": "fingerprint patches -> Transformer",
    "smiles_transformer": "SMILES tokens -> Transformer",
}
DATA = ["prepare_data/policySmilesTrain.pkl.gz", "prepare_data/policySmilesTest.pkl.gz"]


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def write_summary(out_dir, models):
    rows = []
    for m in models:
        path = os.path.join(ROOT, out_dir, f"{m}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            log = json.load(f)
        best = log.get("best")
        if not best:
            continue
        rows.append((m, log, best))

    lines = [
        "# Policy backbone comparison",
        "",
        f"Same subset for every run: {rows[0][1]['n_train']} train / "
        f"{rows[0][1]['n_test']} val reactions, {rows[0][1]['n_rules']} templates."
        if rows else "No results yet.",
        f"Device: {rows[0][1]['device']}"
        + (f" ({rows[0][1]['gpu']})" if rows[0][1].get("gpu") else "")
        + f", epochs: {rows[0][1]['epochs']}, batch: {rows[0][1]['batch_size']}"
        if rows else "",
        "",
        "| model | input | params | best epoch | top-1 | top-10 | top-50 | val loss | s/epoch |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, log, b in rows:
        hist = log["history"]
        sec = hist[-1]["elapsed_s"] / len(hist)
        lines.append(
            f"| {m} | {LABEL[m]} | {log['params']:,} | {b['epoch']} | "
            f"{b['top1']:.4f} | {b['top10']:.4f} | {b['top50']:.4f} | "
            f"{b['val_loss']:.4f} | {sec:.1f} |"
        )

    path = os.path.join(ROOT, out_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="results/backbones")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()

    if not args.summary_only:
        if not all(os.path.exists(os.path.join(ROOT, d)) for d in DATA):
            run([sys.executable, "scripts/build_smiles_policy_data.py"])

        for m in args.models:
            cmd = [sys.executable, "scripts/compare_policy_backbones.py",
                   "--model", m, "--epochs", str(args.epochs),
                   "--batch-size", str(args.batch_size), "--lr", str(args.lr),
                   "--device", args.device, "--out-dir", args.out_dir]
            if args.amp:
                cmd.append("--amp")
            run(cmd)

    write_summary(args.out_dir, args.models)


if __name__ == "__main__":
    main()
