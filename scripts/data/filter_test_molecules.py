"""
Build the extra test sets with the filtering the paper prescribes.

  python scripts/data/filter_test_molecules.py --input data/raw/mol_props/logS.csv \
      --name logS --policy ./saved_model/policy_model.ckpt

The paper removes, from each of the nine property datasets, molecules that make
the benchmark trivially easy (Methods, "Test dataset"):

  1. molecules present in the USPTO database or in the building block set
  2. molecules solved by a heuristic BFS planner within 50 steps
  3. molecules solvable by the one-step model in a single step

Rule 1 needs no model. Rules 2 and 3 call the expansion policy, so they run
only when --policy is given; without it the stage applies rule 1, says that the
rest were skipped, and writes the result anyway so the step is still usable.

Output: test_dataset/<name>.pkl, a list of SMILES — the shape MEEA_PC_parallel
already loads.
"""
import argparse
import csv
import json
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, "rdchiral"))

from corpus import iter_reactions

SMILES_COLUMNS = ("smiles", "canonical_smiles", "isosmiles", "structure", "mol")


def canonical(smiles):
    """Canonical SMILES with atom maps stripped.

    The USPTO corpus is atom-mapped; a mapped SMILES never string-matches an
    unmapped test molecule, so the maps have to go before any comparison.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def read_molecules(path):
    """SMILES out of a csv/smi/txt/sdf file, whichever the data library served."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".sdf":
        from rdkit import Chem

        return [Chem.MolToSmiles(m) for m in Chem.SDMolSupplier(path) if m is not None]

    out = []
    with open(path) as f:
        sample = f.readline()
        f.seek(0)
        if "," in sample or "\t" in sample:
            delimiter = "\t" if "\t" in sample else ","
            reader = csv.DictReader(f, delimiter=delimiter)
            column = next((c for c in reader.fieldnames or []
                           if c and c.strip().lower() in SMILES_COLUMNS), None)
            if column is None:
                sys.exit(f"no SMILES column in {path}; columns: {reader.fieldnames}")
            out = [row[column].strip() for row in reader if row.get(column)]
        else:
            out = [line.split()[0] for line in f if line.strip()]
    return out


def load_known_molecules(building_blocks, reactions):
    known = set()
    if building_blocks and os.path.exists(building_blocks):
        with open(building_blocks) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("mol"):
                    known.add(row["mol"].strip())
        print(f"{len(known):,} building blocks")
    blocks = set(known)

    if reactions:
        before = len(known)
        for r in iter_reactions(reactions):
            for part in (r["product"], *r["reactants"].split(".")):
                c = canonical(part)
                if c:
                    known.add(c)
        print(f"{len(known) - before:,} molecules from the USPTO corpus")
    return known, blocks


def one_step_solved(model, smiles, blocks, topk):
    result = model.run(smiles, topk=topk)
    if not result:
        return False
    for reactants in result["reactants"]:
        if all(part in blocks for part in reactants.split(".")):
            return True
    return False


def bfs_solved(model, smiles, blocks, max_steps, topk):
    """Breadth-first expansion with a budget of `max_steps` one-step calls."""
    frontier = [(smiles,)]
    visited = set()
    calls = 0

    while frontier and calls < max_steps:
        state = frontier.pop(0)
        pending = [m for m in state if m not in blocks]
        if not pending:
            return True
        target = pending[0]
        if target in visited:
            continue
        visited.add(target)

        result = model.run(target, topk=topk)
        calls += 1
        if not result:
            continue
        rest = tuple(m for m in state if m != target)
        for reactants in result["reactants"]:
            frontier.append(rest + tuple(reactants.split(".")))
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="raw dataset file")
    p.add_argument("--name", required=True, help="dataset name, e.g. logS")
    p.add_argument("--building-blocks", default="prepare_data/origin_dict.csv")
    p.add_argument("--reactions", help="USPTO corpus, for rule 1")
    p.add_argument("--policy", help="policy checkpoint; enables rules 2 and 3")
    p.add_argument("--templates", default="./prepare_data/template_rules_1.dat")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--bfs-steps", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="test_dataset")
    args = p.parse_args()

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    raw = read_molecules(args.input)
    molecules, seen = [], set()
    for s in raw:
        c = canonical(s)
        if c and c not in seen:
            seen.add(c)
            molecules.append(c)
    print(f"{len(raw)} rows -> {len(molecules)} unique valid molecules")

    known, blocks = load_known_molecules(args.building_blocks, args.reactions)
    kept = [m for m in molecules if m not in known]
    stats = {"name": args.name, "input": args.input, "unique_molecules": len(molecules),
             "removed_known": len(molecules) - len(kept)}

    if args.policy:
        from policyNet import MLPModel

        model = MLPModel(args.policy, args.templates, device=args.device)
        survivors = []
        removed_one_step = removed_bfs = 0
        for i, smiles in enumerate(kept, 1):
            if one_step_solved(model, smiles, blocks, args.topk):
                removed_one_step += 1
            elif bfs_solved(model, smiles, blocks, args.bfs_steps, min(args.topk, 10)):
                removed_bfs += 1
            else:
                survivors.append(smiles)
            if i % 100 == 0:
                print(f"  {i}/{len(kept)} screened, {len(survivors)} kept", flush=True)
        kept = survivors
        stats["removed_one_step"] = removed_one_step
        stats["removed_bfs"] = removed_bfs
    else:
        stats["skipped"] = "rules 2 and 3 (no --policy given)"
        print("no --policy: keeping molecules that a planner may still solve trivially")

    stats["kept"] = len(kept)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.name}.pkl")
    with open(out, "wb") as f:
        pickle.dump(kept, f)
    with open(os.path.join(args.out_dir, f"{args.name}_filter_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
