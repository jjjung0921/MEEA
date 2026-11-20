# policyNet backbone benchmark on the RTX 4060 Ti desktop

Compares three policy backbones on identical data: the fingerprint MLP of the
paper, the fingerprint-patch Transformer from slides 14–16, and a Transformer
over SMILES tokens — the representation slide 15 points to ("SMILES와 같이 위치
순서가 상관 있는 모델에 더 적합할 것으로 예상").

## 1. Environment (once)

```bash
conda create -n meea-gpu python=3.10 -y && conda activate meea-gpu
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
pip install -e rdchiral
```

Check the GPU is visible:

```bash
python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"
```

## 2. Data

The benchmark needs three files that are not in git (`prepare_data/` is
ignored). Copy them from the Mac checkout — 3 MB in total:

```bash
scp mac:~/projects/_archive/MEEA/prepare_data/policySmiles{Train,Test}.pkl.gz prepare_data/
scp mac:~/projects/_archive/MEEA/prepare_data/uspto_template.pkl.gz prepare_data/
```

Alternatively regenerate the SMILES files on the desktop from the original
policy pickles plus `data/uspto50k.tab`:

```bash
python scripts/build_smiles_policy_data.py
```

It recovers the SMILES behind every fingerprint row by re-fingerprinting the
products of `uspto50k.tab` and matching the packed bits, so the split and the
labels stay exactly those of `policyTrain/Test.pkl.gz`. 4,352 train and 1,095
val rows have no match (they come from a larger USPTO dump than the 50k table)
and are dropped, leaving **35,634 train / 8,903 val** reactions over 14,407
templates. Every backbone is trained on that same subset, so the comparison is
apples to apples — but the numbers are not directly comparable to the slides,
which used the full set and 100 epochs.

## 3. Run

```bash
python scripts/run_backbone_benchmark.py --epochs 30 --batch-size 512 --amp
```

That builds the data if needed, trains the three backbones one after another,
and writes `results/backbones/summary.md`. One backbone only:

```bash
python scripts/compare_policy_backbones.py --model smiles_transformer \
    --epochs 30 --batch-size 512 --amp
```

Useful flags: `--lr`, `--max-len` (SMILES truncation, default 160 tokens),
`--num-workers`, `--device cuda:0`, `--out-dir`.

8 GB of VRAM is plenty at batch 512; the largest model is 5.8 M parameters and
the 14,407-way head dominates memory, not the sequence.

## 4. Output

Per model, in `results/backbones/`:

- `<model>.json` — config plus per-epoch `train_loss`, `val_loss`, `top1`,
  `top10`, `top50`, `elapsed_s`, and the best epoch under `best`
- `<model>.ckpt` — weights of the best top-1 epoch
- `smiles_transformer_vocab.json` — token vocabulary (needed for inference)
- `summary.md` — the comparison table

Rebuild the table without retraining:

```bash
python scripts/run_backbone_benchmark.py --summary-only
```

## 5. Mac reference numbers (CPU, same subset)

Partial run before the GPU move, for sanity-checking the desktop results:

| model | epochs run | top-1 | top-10 | top-50 | val loss | s/epoch |
|---|---|---|---|---|---|---|
| mlp | 21 (best 18) | 0.3454 | 0.6131 | 0.6916 | 6.0902 | 6.5 |
| smiles_transformer | 1 | 0.0543 | 0.1869 | 0.3452 | 7.7452 | 246 |

MPS is not usable here: torch 1.13 crashes with SIGBUS inside the Transformer
encoder, which is why the Mac runs are CPU-only and the real comparison belongs
on the desktop.
