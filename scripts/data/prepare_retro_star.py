"""
Unpack the Retro* data package and put its pieces where this repo expects them.

  python scripts/data/prepare_retro_star.py

Reads data/raw/retro_data.zip (see fetch_sources.py), extracts it to
data/raw/retro_star/, then reports what it found and copies the files the rest
of the pipeline and the search code read:

  building blocks (a csv with a `mol` column)  -> prepare_data/origin_dict.csv
  routes_possible_test_hard.pkl                -> prepare_data/
  one-step model checkpoint                    -> left in place, path reported

Nothing is guessed: files are matched by name, and whatever is not found is
listed so it can be pointed at explicitly with --building-blocks / --routes.
"""
import argparse
import json
import os
import shutil
import sys
import zipfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from corpus import find_reaction_files

RAW_ZIP = "data/raw/retro_data.zip"
EXTRACT_DIR = "data/raw/retro_star"
REPORT = "data/raw/retro_star_layout.json"


def find_one(root, *name_parts, suffixes=()):
    """First file under root whose name contains all name_parts."""
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            low = name.lower()
            if all(part in low for part in name_parts) and (
                not suffixes or low.endswith(suffixes)
            ):
                return os.path.join(dirpath, name)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default=RAW_ZIP)
    p.add_argument("--extract-dir", default=EXTRACT_DIR)
    p.add_argument("--building-blocks", help="override the detected building block file")
    p.add_argument("--routes", help="override the detected route file")
    p.add_argument("--skip-extract", action="store_true")
    args = p.parse_args()

    if not args.skip_extract:
        if not os.path.exists(args.zip):
            sys.exit(f"{args.zip} not found — run scripts/data/fetch_sources.py --list")
        os.makedirs(args.extract_dir, exist_ok=True)
        print(f"extracting {args.zip} -> {args.extract_dir}")
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(args.extract_dir)

    root = args.extract_dir
    if not os.path.isdir(root):
        sys.exit(f"{root} does not exist")

    blocks = args.building_blocks or find_one(root, "origin_dict", suffixes=(".csv",)) \
        or find_one(root, "building_block", suffixes=(".csv", ".pkl"))
    routes = args.routes or find_one(root, "routes_possible", suffixes=(".pkl",)) \
        or find_one(root, "routes", suffixes=(".pkl",))
    model = find_one(root, "one_step", suffixes=(".ckpt", ".pt", ".pkl")) \
        or find_one(root, "policy", suffixes=(".ckpt", ".pt"))
    templates = find_one(root, "template", suffixes=(".csv", ".dat", ".txt", ".pkl", ".gz"))
    reactions = find_reaction_files(root)

    layout = {
        "extract_dir": root,
        "building_blocks": blocks,
        "routes": routes,
        "one_step_model": model,
        "templates": templates,
        "reaction_files": reactions[:20],
    }
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(layout, f, indent=2)
    print(json.dumps(layout, indent=2))

    os.makedirs("prepare_data", exist_ok=True)
    if blocks and blocks.endswith(".csv"):
        shutil.copy(blocks, "prepare_data/origin_dict.csv")
        print("copied -> prepare_data/origin_dict.csv")
    elif blocks:
        print(f"building blocks are not a csv ({blocks}); "
              "run scripts/data/build_building_blocks.py instead")
    else:
        print("building blocks NOT found — pass --building-blocks or build them "
              "from eMolecules with scripts/data/build_building_blocks.py")

    if routes:
        shutil.copy(routes, "prepare_data/routes_possible_test_hard.pkl")
        print("copied -> prepare_data/routes_possible_test_hard.pkl")
    else:
        print("test routes NOT found — pass --routes")

    if not reactions:
        print("no reaction corpus found in the package; point the template and "
              "policy stages at one with --reactions")
    print(f"\nlayout written to {REPORT}")


if __name__ == "__main__":
    main()
