"""
Utility to build policyNet training data from USPTO reaction templates.

This script expects the outputs of rdchiral template extraction:
- data/uspto.reactions.json.gz
- data/uspto.templates.json.gz

It produces three artifacts under ./prepare_data/:
- uspto_template.pkl.gz           : list of unique reaction SMARTS templates
- policyTrain.pkl.gz              : {'smiles': packed_fps, 'template': labels}
- policyTest.pkl.gz               : same structure for held-out split
"""
import argparse
import gzip
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def pack_morgan_fp(smiles: str, fp_dim: int = 2048) -> np.ndarray:
    """Create a packed Morgan fingerprint (radius 2, chirality on)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=fp_dim, includeChirality=True
    )
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(fp.GetNumBits(), dtype=np.uint8)
    arr[list(fp.GetOnBits())] = 1
    return np.packbits(arr)


def load_reactions(path: str) -> Dict[int, Dict]:
    with gzip.open(path, "rt") as f:
        reactions = json.load(f)
    return {r["_id"]: r for r in reactions}


def load_templates(path: str) -> Dict[int, Dict]:
    with gzip.open(path, "rt") as f:
        templates = json.load(f)
    return {t["reaction_id"]: t for t in templates if "reaction_smarts" in t}


def build_dataset(
    reactions_path: str, templates_path: str, fp_dim: int
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    reactions = load_reactions(reactions_path)
    templates = load_templates(templates_path)

    fps: List[np.ndarray] = []
    labels: List[int] = []
    template_list: List[str] = []
    template_to_idx: Dict[str, int] = {}

    for rxn_id, rxn in tqdm(reactions.items(), desc="processing reactions"):
        tmpl = templates.get(rxn_id)
        if tmpl is None:
            continue
        template = tmpl["reaction_smarts"]
        product = rxn.get("products")
        if not product:
            continue
        try:
            fp = pack_morgan_fp(product, fp_dim=fp_dim)
        except Exception:
            continue

        if template not in template_to_idx:
            template_to_idx[template] = len(template_list)
            template_list.append(template)

        fps.append(fp)
        labels.append(template_to_idx[template])

    return fps, labels, template_list


def save_policy_split(
    fps: List[np.ndarray],
    labels: List[int],
    template_list: List[str],
    output_dir: str,
    test_size: float = 0.1,
):
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        fps, labels, test_size=test_size, random_state=42, stratify=labels
    )

    def dump(path: str, X: List[np.ndarray], y: List[int]):
        payload = {"smiles": np.array(X, dtype=np.uint8), "template": np.array(y, dtype=np.int64)}
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f)

    dump(os.path.join(output_dir, "policyTrain.pkl.gz"), X_train, y_train)
    dump(os.path.join(output_dir, "policyTest.pkl.gz"), X_test, y_test)

    with gzip.open(os.path.join(output_dir, "uspto_template.pkl.gz"), "wb") as f:
        pickle.dump(template_list, f)


def main():
    parser = argparse.ArgumentParser(description="Build policyNet datasets from USPTO templates")
    parser.add_argument(
        "--reactions",
        default="rdchiral/templates/data/uspto.reactions.json.gz",
        help="Path to uspto.reactions.json.gz",
    )
    parser.add_argument(
        "--templates",
        default="rdchiral/templates/data/uspto.templates.json.gz",
        help="Path to uspto.templates.json.gz",
    )
    parser.add_argument("--fp-dim", type=int, default=2048, help="Fingerprint dimension")
    parser.add_argument(
        "--output-dir", default="prepare_data", help="Where to write pkl.gz outputs"
    )
    parser.add_argument("--test-size", type=float, default=0.1, help="Test split ratio")
    args = parser.parse_args()

    fps, labels, template_list = build_dataset(args.reactions, args.templates, args.fp_dim)
    save_policy_split(fps, labels, template_list, args.output_dir, test_size=args.test_size)
    print(
        f"Done. Templates: {len(template_list)}, train samples: {len(labels) - int(len(labels)*args.test_size)}, "
        f"test samples: {int(len(labels)*args.test_size)}"
    )


if __name__ == "__main__":
    main()
