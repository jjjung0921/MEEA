"""
Extract retrosynthesis templates from the USPTO corpus with rdchiral.

  python scripts/data/extract_templates.py --reactions <file> [--min-count 1]

Templates are extracted from the TRAINING split only — a template mined from a
validation or test reaction would leak that reaction into the policy's label
set. Reactions whose split the corpus does not state are treated as training
data, and the stage says so.

Outputs:
  prepare_data/uspto_template.pkl.gz   list[str] of reaction SMARTS, the label
                                       space of the policy network
  prepare_data/template_stats.json     counts, failures, filtering summary

Requires atom-mapped reactions, which is what the USPTO dumps provide. If
mapping is missing, extraction fails for every reaction and the stage says so
rather than writing an empty template list.
"""
import argparse
import gzip
import json
import os
import pickle
import sys
from collections import Counter
from multiprocessing import Pool

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_ROOT, "rdchiral"))

from corpus import iter_reactions

TRAIN_SPLITS = {"train", "training", None, ""}


def extract_one(reaction):
    from rdchiral.template_extractor import extract_from_reaction

    try:
        out = extract_from_reaction({
            "reactants": reaction["reactants"],
            "products": reaction["product"],
            "_id": 0,
        })
    except Exception:
        return None
    return out.get("reaction_smarts") if out else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reactions", required=True, help="USPTO corpus file")
    p.add_argument("--min-count", type=int, default=1,
                   help="drop templates seen fewer than this many times")
    p.add_argument("--max-reactions", type=int, default=0, help="debug cap")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default="prepare_data/uspto_template.pkl.gz")
    p.add_argument("--stats", default="prepare_data/template_stats.json")
    args = p.parse_args()

    reactions, skipped_split = [], 0
    unknown_split = 0
    for r in iter_reactions(args.reactions):
        split = (r.get("split") or "").lower() or None
        if split not in TRAIN_SPLITS:
            skipped_split += 1
            continue
        if split is None:
            unknown_split += 1
        reactions.append(r)
        if args.max_reactions and len(reactions) >= args.max_reactions:
            break

    print(f"{len(reactions)} training reactions "
          f"({skipped_split} held out by split, {unknown_split} with no split label)")
    if unknown_split == len(reactions) and reactions:
        print("NOTE: this corpus states no split, so every reaction is treated as training data")

    with Pool(args.workers) as pool:
        smarts = pool.map(extract_one, reactions, chunksize=200)

    counts = Counter(s for s in smarts if s)
    failed = sum(1 for s in smarts if not s)
    if not counts:
        sys.exit("no template could be extracted — are the reactions atom-mapped? "
                 "(see scripts/map_and_build_policy_from_tdc.py for mapping with RXNMapper)")

    kept = [s for s, c in counts.most_common() if c >= args.min_count]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.open(args.out, "wb") as f:
        pickle.dump(kept, f)

    stats = {
        "reactions_read": len(reactions),
        "held_out_by_split": skipped_split,
        "extraction_failed": failed,
        "unique_templates": len(counts),
        "min_count": args.min_count,
        "templates_kept": len(kept),
        "coverage_of_training_reactions": round(
            sum(counts[s] for s in kept) / max(1, len(reactions)), 4),
        "source": args.reactions,
    }
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
