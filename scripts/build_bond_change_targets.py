"""Build shared bond-change labels without changing the policy benchmark splits."""
import argparse
import csv
import gzip
import json
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions

sys.path.append(str(Path(__file__).resolve().parents[1] / "rdchiral"))
from rdchiral.template_extractor import extract_from_reaction

RDLogger.DisableLog("rdApp.*")


def mapped_bonds(molecules):
    result = {}
    for mol in molecules:
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtom(), bond.GetEndAtom()
            if a.GetAtomMapNum() and b.GetAtomMapNum():
                key = tuple(sorted((a.GetAtomMapNum(), b.GetAtomMapNum())))
                result[key] = (tuple(sorted((a.GetAtomicNum(), b.GetAtomicNum()))),
                               str(bond.GetBondType()))
    return result


def features(smarts):
    reaction = rdChemReactions.ReactionFromSmarts(smarts)
    before = mapped_bonds(reaction.GetReactants())
    after = mapped_bonds(reaction.GetProducts())
    edits = set()
    for key in before.keys() | after.keys():
        atoms = (before.get(key) or after[key])[0]
        old = before[key][1] if key in before else "NONE"
        new = after[key][1] if key in after else "NONE"
        if old != new:
            edits.add(f"bond:{atoms[0]}-{atoms[1]}:{old}>{new}")
    # Use a coarse joint target for the number of precursor fragments in the rule.
    precursors = sum(len(Chem.GetMolFrags(mol)) for mol in reaction.GetProducts())
    edits.add(f"precursor_count:{min(precursors, 4)}")
    return edits


def load(path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def source_key(rxn):
    reactants, _, product = rxn.split(">")
    try:
        rule = extract_from_reaction({"reactants": reactants, "products": product, "_id": 0})
        return canonical(product), rule.get("reaction_smarts")
    except (ValueError, KeyError, AttributeError, RuntimeError):
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("prepare_data/bond_change_targets.npz"))
    parser.add_argument("--source", type=Path, default=Path("data/raw/uspto50k_figshare"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rules = load("prepare_data/uspto_template.pkl.gz")
    train = load("prepare_data/policySmilesTrain.pkl.gz")
    val = load("prepare_data/policySmilesTest.pkl.gz")
    y = np.asarray(train["template"])
    by_rule = {r: features(rules[r]) for r in np.unique(y)}
    vocabulary = sorted(set().union(*by_rule.values()))
    indices = {name: i for i, name in enumerate(vocabulary)}
    targets = np.zeros((len(y), len(vocabulary)), dtype=bool)
    for i, rule in enumerate(y):
        targets[i, [indices[name] for name in by_rule[rule]]] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, targets=targets, labels=y, vocabulary=vocabulary)

    with ProcessPoolExecutor(args.workers) as pool:
        source = set()
        for name in ("raw_train.csv", "raw_val.csv", "raw_test.csv"):
            with (args.source / name).open() as stream:
                reactions = (row["reactants>reagents>production"] for row in csv.DictReader(stream))
                source.update((product, rule) for product, rule in
                              pool.map(source_key, reactions, chunksize=128) if product and rule)
    support = targets.sum(axis=0)
    counts = Counter(y)
    singleton = np.asarray([counts[r] == 1 for r in y])
    shared_edit = np.asarray([name.startswith("bond:") for name in vocabulary]) & (support > 1)
    def matched(data):
        return sum((canonical(s), rules[r]) in source
                   for s, r in zip(data["smiles"], data["template"]))
    stats = {
        "train_rows": len(y), "new_train_rows": 0,
        "exact_source_matches_train": matched(train), "exact_source_matches_val": matched(val),
        "rows_with_bond_edits": int(targets[:, [s.startswith("bond:") for s in vocabulary]].any(axis=1).sum()),
        "bond_edit_assignments": int(support[[s.startswith("bond:") for s in vocabulary]].sum()),
        "auxiliary_positive_assignments": int(support.sum()),
        "bond_edit_types": sum(s.startswith("bond:") for s in vocabulary),
        "precursor_count_types": sum(s.startswith("precursor_count:") for s in vocabulary),
        "singleton_template_rows": int(singleton.sum()),
        "singleton_rows_with_a_shared_bond_edit": int(targets[singleton][:, shared_edit].any(axis=1).sum()),
    }
    args.out.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
