"""
Reading a USPTO reaction corpus, whatever shape it arrives in.

The Retro* package, the rdchiral extraction output and the old uspto50k.tab in
this repo all carry the same information in different containers, so every
stage of the pipeline goes through one reader instead of hardcoding a layout.

A reaction is a dict: {"reactants": str, "product": str, "split": str|None}.
`split` is kept when the source has one (Retro*'s official train/val/test), so
templates are never extracted from reactions the model is tested on.
"""
import csv
import gzip
import json
import os
from typing import Dict, Iterator, List, Optional

REACTION_KEYS = ("rxn_smiles", "reaction_smiles", "reaction", "rxn")
SPLIT_KEYS = ("split", "set", "dataset")


def _split_rxn(rxn: str):
    parts = rxn.strip().split(">")
    if len(parts) == 3:  # reactants>reagents>products
        reactants, _, product = parts
    elif len(parts) == 2:
        reactants, product = parts
    else:
        return None
    reactants, product = reactants.strip(), product.strip()
    if not reactants or not product:
        return None
    return reactants, product


def _from_row(row: Dict[str, str]) -> Optional[Dict]:
    lower = {k.lower().strip(): (v or "").strip().strip('"') for k, v in row.items() if k}
    split = next((lower[k] for k in SPLIT_KEYS if lower.get(k)), None)

    for key in REACTION_KEYS:
        if lower.get(key):
            parsed = _split_rxn(lower[key])
            if parsed:
                return {"reactants": parsed[0], "product": parsed[1], "split": split}
            return None

    reactants = lower.get("reactants") or lower.get("reactant")
    product = lower.get("products") or lower.get("product")
    if reactants and product:
        return {"reactants": reactants, "product": product, "split": split}
    return None


def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def iter_reactions(path: str) -> Iterator[Dict]:
    """Yield reactions from a csv/tab/json/jsonl/txt file, gzipped or not."""
    base = path[:-3] if path.endswith(".gz") else path
    ext = os.path.splitext(base)[1].lower()

    with _open(path) as f:
        if ext == ".json":
            payload = json.load(f)
            rows = payload if isinstance(payload, list) else payload.get("reactions", [])
            for row in rows:
                parsed = _from_row(row) if isinstance(row, dict) else None
                if parsed:
                    yield parsed
            return

        if ext == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    parsed = _from_row(json.loads(line))
                    if parsed:
                        yield parsed
            return

        if ext in (".csv", ".tab", ".tsv", ".txt"):
            sample = f.read(4096)
            f.seek(0)
            if ">" in sample and "," not in sample.split("\n")[0] and "\t" not in sample.split("\n")[0]:
                for line in f:  # bare "reactants>>product" lines
                    parsed = _split_rxn(line)
                    if parsed:
                        yield {"reactants": parsed[0], "product": parsed[1], "split": None}
                return
            delimiter = "\t" if "\t" in sample.split("\n")[0] else ","
            for row in csv.DictReader(f, delimiter=delimiter):
                parsed = _from_row(row)
                if parsed:
                    yield parsed
            return

    raise ValueError(f"unsupported reaction file: {path}")


def find_reaction_files(root: str) -> List[str]:
    """Files under `root` that look like a reaction corpus, largest first."""
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            low = name.lower()
            if not any(low.endswith(e) for e in
                       (".csv", ".tab", ".tsv", ".txt", ".json", ".jsonl",
                        ".csv.gz", ".json.gz", ".jsonl.gz", ".txt.gz")):
                continue
            if not any(tag in low for tag in ("rxn", "reaction", "uspto", "train", "valid", "test")):
                continue
            path = os.path.join(dirpath, name)
            hits.append((os.path.getsize(path), path))
    return [p for _, p in sorted(hits, reverse=True)]
