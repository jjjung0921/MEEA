"""
Turn the eMolecules dump into the building block file the search reads.

  python scripts/data/build_building_blocks.py

Input  data/raw/emolecules_<snapshot>.smi.gz — lines of "SMILES VERSION_ID PARENT_ID"
Output prepare_data/origin_dict.csv — columns (,mol,idx), which is what
       MEEA_PC_parallel.prepare_starting_molecules reads as `pd.read_csv(...)['mol']`

SMILES are canonicalised with RDKit before they are written, because the search
decides that a molecule is available by string equality against this set: an
uncanonicalised building block is a building block the planner can never match.
Duplicates after canonicalisation are dropped and counted.
"""
import argparse
import csv
import gzip
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sources import EMOLECULES_SNAPSHOT

DEFAULT_IN = f"data/raw/emolecules_{EMOLECULES_SNAPSHOT}.smi.gz"


def canonical(smiles, max_atoms):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if max_atoms and mol.GetNumHeavyAtoms() > max_atoms:
        return None
    return Chem.MolToSmiles(mol)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_IN)
    p.add_argument("--out", default="prepare_data/origin_dict.csv")
    p.add_argument("--max-atoms", type=int, default=0,
                   help="drop building blocks larger than this many heavy atoms (0 = keep all)")
    p.add_argument("--max-lines", type=int, default=0, help="debug cap")
    args = p.parse_args()

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    if not os.path.exists(args.input):
        sys.exit(f"{args.input} not found — run scripts/data/fetch_sources.py --only emolecules")

    seen = set()
    read = invalid = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with gzip.open(args.input, "rt") as fin, open(args.out, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["", "mol", "idx"])

        for line in fin:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if fields[0].lower() in ("smiles", "isosmiles"):   # header line
                continue
            read += 1
            smiles = canonical(fields[0], args.max_atoms)
            if smiles is None:
                invalid += 1
            elif smiles not in seen:
                seen.add(smiles)
                writer.writerow([len(seen) - 1, smiles, fields[1] if len(fields) > 1 else ""])
            if read % 1_000_000 == 0:
                print(f"  {read:,} read, {len(seen):,} kept", flush=True)
            if args.max_lines and read >= args.max_lines:
                break

    stats = {"input": args.input, "lines_read": read, "unparsable": invalid,
             "building_blocks": len(seen), "max_atoms": args.max_atoms}
    stats_path = os.path.join(os.path.dirname(args.out) or ".", "building_blocks_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
