"""
Build the policy network's training data from the USPTO corpus.

  python scripts/data/build_policy_dataset.py --reactions <file>

For every reaction it extracts the template, looks the template up in the label
space written by extract_templates.py, and stores the product in BOTH
representations the repo now trains on:

  prepare_data/policyTrain.pkl.gz        {'smiles': packed Morgan fp, 'template': label}
  prepare_data/policyTest.pkl.gz         same, validation split
  prepare_data/policySmilesTrain.pkl.gz  {'smiles': [SMILES], 'fp': packed, 'template': label}
  prepare_data/policySmilesTest.pkl.gz   same, validation split

The fingerprint files keep the schema policyNet.py already reads, so the paper's
MLP is unaffected. The SMILES files carry the same rows, which is what makes the
fingerprint/SMILES backbone comparison a controlled one — the earlier
scripts/build_smiles_policy_data.py recovered them after the fact by matching
fingerprints and lost 11% of the rows; this stage never needs that.

The split comes from the corpus when it states one (Retro*'s official split),
otherwise rows are split by a hash of the product SMILES so the assignment is
stable across runs and never puts the same molecule on both sides.
"""
import argparse
import gzip
import hashlib
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_ROOT, "rdchiral"))

from corpus import iter_reactions

FP_DIM = 2048
VAL_SPLITS = {"val", "valid", "validation", "test"}


def _worker(reaction):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdchiral.template_extractor import extract_from_reaction

    RDLogger.DisableLog("rdApp.*")

    try:
        out = extract_from_reaction({
            "reactants": reaction["reactants"],
            "products": reaction["product"],
            "_id": 0,
        })
        smarts = out.get("reaction_smarts") if out else None
    except Exception:
        smarts = None
    if not smarts:
        return None

    # the product carries atom maps in the corpus; strip them for the model input
    mol = Chem.MolFromSmiles(reaction["product"])
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(mol)

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=FP_DIM, includeChirality=True)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(fp.GetNumBits(), dtype=np.uint8)
    arr[list(fp.GetOnBits())] = 1
    return smiles, np.packbits(arr), smarts, (reaction.get("split") or "").lower()


def is_val(split, smiles, val_fraction):
    if split in VAL_SPLITS:
        return True
    if split:
        return False
    digest = hashlib.sha256(smiles.encode()).digest()
    return (int.from_bytes(digest[:4], "big") % 10000) < val_fraction * 10000


def dump(path, payload):
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reactions", required=True)
    p.add_argument("--templates", default="prepare_data/uspto_template.pkl.gz")
    p.add_argument("--out-dir", default="prepare_data")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="used only when the corpus states no split")
    p.add_argument("--max-reactions", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()

    with gzip.open(args.templates, "rb") as f:
        templates = pickle.load(f)
    template_to_idx = {t: i for i, t in enumerate(templates)}
    print(f"{len(templates)} templates in the label space")

    reactions = []
    for r in iter_reactions(args.reactions):
        reactions.append(r)
        if args.max_reactions and len(reactions) >= args.max_reactions:
            break
    print(f"{len(reactions)} reactions read")

    with Pool(args.workers) as pool:
        rows = pool.map(_worker, reactions, chunksize=200)

    out = {"train": {"smiles": [], "fp": [], "template": []},
           "val": {"smiles": [], "fp": [], "template": []}}
    failed = unknown_template = 0

    for row in rows:
        if row is None:
            failed += 1
            continue
        smiles, packed, smarts, split = row
        label = template_to_idx.get(smarts)
        if label is None:          # filtered out by --min-count, or a test-only template
            unknown_template += 1
            continue
        bucket = out["val" if is_val(split, smiles, args.val_fraction) else "train"]
        bucket["smiles"].append(smiles)
        bucket["fp"].append(packed)
        bucket["template"].append(label)

    os.makedirs(args.out_dir, exist_ok=True)
    stats = {"reactions_read": len(reactions), "extraction_failed": failed,
             "template_not_in_label_space": unknown_template,
             "n_templates": len(templates), "source": args.reactions}

    for name, key in (("Train", "train"), ("Test", "val")):
        bucket = out[key]
        fps = np.asarray(bucket["fp"], dtype=np.uint8)
        labels = np.asarray(bucket["template"], dtype=np.int64)
        dump(os.path.join(args.out_dir, f"policy{name}.pkl.gz"),
             {"smiles": fps, "template": labels})           # schema policyNet.py reads
        dump(os.path.join(args.out_dir, f"policySmiles{name}.pkl.gz"),
             {"smiles": bucket["smiles"], "fp": fps, "template": labels})
        stats[f"n_{key}"] = len(labels)

    with open(os.path.join(args.out_dir, "policy_dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
