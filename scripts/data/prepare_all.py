"""
Run the whole data pipeline in order.

  python scripts/data/prepare_all.py --reactions data/raw/retro_star/dataset/<corpus file>

Stages:
  1 fetch       check the raw sources are on disk (downloads what it can)
  2 retro_star  unpack the Retro* package, place building blocks and test routes
  3 templates   extract the template label space from the training split
  4 policy      build the policy training data (fingerprint + SMILES)
  5 blocks      build prepare_data/origin_dict.csv from eMolecules
  6 testsets    filter the nine property datasets

Use --only to run a subset, e.g. `--only templates policy`. Stage 2 prints
where the reaction corpus ended up; pass that path as --reactions.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = "scripts/data"
STAGES = ["fetch", "retro_star", "templates", "policy", "blocks", "testsets"]
MOL_PROPS = ["logS", "logP", "LD50", "Ames", "BBBP", "DPP4", "BACE", "SVS", "ClinTox"]


def run(args, allow_fail=False):
    print("+", " ".join(args), flush=True)
    result = subprocess.run([sys.executable] + args, cwd=ROOT)
    if result.returncode and not allow_fail:
        sys.exit(result.returncode)
    return result.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reactions", help="USPTO corpus file (needed by templates/policy)")
    p.add_argument("--only", nargs="+", choices=STAGES, default=STAGES)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--policy", help="policy checkpoint, enables the full test-set filtering")
    p.add_argument("--mol-props-dir", default="data/raw/mol_props")
    args = p.parse_args()

    if "fetch" in args.only:
        run([f"{HERE}/fetch_sources.py", "--check"], allow_fail=True)

    if "retro_star" in args.only:
        run([f"{HERE}/prepare_retro_star.py"], allow_fail=True)

    needs_corpus = {"templates", "policy"} & set(args.only)
    if needs_corpus and not args.reactions:
        sys.exit("--reactions is required for the templates and policy stages "
                 "(see data/raw/retro_star_layout.json for candidates)")

    if "templates" in args.only:
        run([f"{HERE}/extract_templates.py", "--reactions", args.reactions,
             "--min-count", str(args.min_count)])

    if "policy" in args.only:
        run([f"{HERE}/build_policy_dataset.py", "--reactions", args.reactions])

    if "blocks" in args.only:
        run([f"{HERE}/build_building_blocks.py"], allow_fail=True)

    if "testsets" in args.only:
        for name in MOL_PROPS:
            candidates = [f for f in os.listdir(os.path.join(ROOT, args.mol_props_dir))
                          if f.lower().startswith(name.lower())] \
                if os.path.isdir(os.path.join(ROOT, args.mol_props_dir)) else []
            if not candidates:
                print(f"[{name}] no file in {args.mol_props_dir}, skipped")
                continue
            cmd = [f"{HERE}/filter_test_molecules.py",
                   "--input", os.path.join(args.mol_props_dir, candidates[0]),
                   "--name", name]
            if args.reactions:
                cmd += ["--reactions", args.reactions]
            if args.policy:
                cmd += ["--policy", args.policy]
            run(cmd, allow_fail=True)

    print("\ndone")


if __name__ == "__main__":
    main()
