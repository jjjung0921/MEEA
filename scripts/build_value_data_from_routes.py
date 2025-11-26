"""
Lightweight utility to fabricate valueEnsemble training data from simple reaction routes.

Input
-----
- A pickle file containing a list of reaction strings in the form "product>>reactant1.reactant2..."
  (default: prepare_data/routes_possible_test_hard.pkl)

Output
------
Files are written under ./data/ and ./val_dataset/ to match valueEnsemble.py defaults:
- data/train_consistency.pkl
- data/val_consistency.pkl
- data/train_fitting.pkl
- data/test_fitting.pkl
- val_dataset/val_fitting.pkl

Notes
-----
The resulting values/costs are heuristic (cost=1.0 per reaction, target_value=#reactants).
They are intended as placeholders; replace with domain-specific labels if available.
"""
import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def pack_fp(smiles: str, fp_dim: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles}")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_dim)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(fp.GetNumBits(), dtype=np.uint8)
    arr[list(fp.GetOnBits())] = 1
    return np.packbits(arr)


def parse_route(route: str) -> Tuple[str, List[str]]:
    if ">>" not in route:
        raise ValueError(f"Route missing '>>': {route}")
    product, reactants = route.split(">>")
    reactant_list = [r for r in reactants.split(".") if r]
    return product, reactant_list


def build_consistency(routes: List[str], fp_dim: int):
    reaction_costs = []
    target_values = []
    reactant_fps = []
    reactant_masks = []

    for route in tqdm(routes, desc="consistency routes"):
        try:
            product, reactants = parse_route(route)
        except ValueError:
            continue
        if not reactants:
            continue

        # padding to 3 reactants max (ValueEnsemble uses [:3])
        fps = np.zeros((3, fp_dim), dtype=np.float32)
        masks = np.zeros(3, dtype=np.float32)
        for i, r in enumerate(reactants[:3]):
            try:
                fps[i, :] = np.unpackbits(pack_fp(r, fp_dim=fp_dim)).astype(np.float32)
                masks[i] = 1.0
            except Exception:
                masks[i] = 0.0

        reactant_fps.append(fps)
        reactant_masks.append(masks)
        reaction_costs.append(1.0)  # placeholder cost
        target_values.append(float(len(reactants)))  # placeholder target

    return {
        "reaction_costs": reaction_costs,
        "target_values": target_values,
        "reactant_fps": reactant_fps,
        "reactant_masks": reactant_masks,
    }


def build_fitting(routes: List[str], fp_dim: int):
    fps = []
    values = []
    masks = []

    for route in tqdm(routes, desc="fitting routes"):
        try:
            product, reactants = parse_route(route)
        except ValueError:
            continue
        try:
            p_fp = np.unpackbits(pack_fp(product, fp_dim=fp_dim)).astype(np.float32)
        except Exception:
            continue

        fps.append(p_fp)
        values.append(float(len(reactants)))  # heuristic value
        masks.append(np.array([1.0] + [0.0] * 4, dtype=np.float32))  # single molecule mask

    return {"fps": fps, "values": values, "masks": masks}


def dump_pickle(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def main():
    parser = argparse.ArgumentParser(description="Build placeholder valueEnsemble data from routes")
    parser.add_argument(
        "--routes",
        default="prepare_data/routes_possible_test_hard.pkl",
        help="Pickle file containing list of 'product>>reactants' strings",
    )
    parser.add_argument("--fp-dim", type=int, default=2048)
    parser.add_argument("--test-size", type=float, default=0.1, help="Split fraction for fitting data")
    args = parser.parse_args()

    with open(args.routes, "rb") as f:
        raw_routes = pickle.load(f)

    routes: List[str] = []
    for entry in raw_routes:
        if isinstance(entry, list):
            routes.extend([r for r in entry if isinstance(r, str)])
        elif isinstance(entry, str):
            routes.append(entry)
    print(f"Loaded {len(routes)} route steps")

    # Consistency split (train/val)
    train_routes, val_routes = train_test_split(routes, test_size=args.test_size, random_state=42)
    train_consistency = build_consistency(train_routes, args.fp_dim)
    val_consistency = build_consistency(val_routes, args.fp_dim)
    dump_pickle("data/train_consistency.pkl", train_consistency)
    dump_pickle("data/val_consistency.pkl", val_consistency)

    # Fitting split (train/test/val)
    fitting_data = build_fitting(routes, args.fp_dim)
    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        fitting_data["fps"],
        fitting_data["values"],
        fitting_data["masks"],
        test_size=args.test_size,
        random_state=42,
        stratify=None,
    )
    train_fit = {"fps": X_train, "values": y_train, "masks": m_train}
    test_fit = {"fps": X_test, "values": y_test, "masks": m_test}
    dump_pickle("data/train_fitting.pkl", train_fit)
    dump_pickle("data/test_fitting.pkl", test_fit)
    dump_pickle("val_dataset/val_fitting.pkl", test_fit)

    print("Data written to data/ and val_dataset/. NOTE: costs/values are heuristic placeholders.")


if __name__ == "__main__":
    main()
