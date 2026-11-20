"""
Where the data of the paper comes from, and how to put it on disk.

The paper (Commun Chem 7:52, Data availability) names three sources:

  retro_star   USPTO reactions, the train/val/test split, the test benchmark
               routes and the one-step model that MEEA* starts from. The
               MEEA*/Retro* lineage uses this split, so templates and policy
               data must be built from it rather than from USPTO-50k.
  emolecules   Commercially available molecules used as building blocks.
  mol_props    The nine extra test sets (logS, logP, LD50, Ames, BBBP, DPP4,
               BACE, SVS, ClinTox) from the MSU 2D data library.

Only `emolecules` has a stable direct URL. The Retro* package lives on Dropbox
and the MSU library is a browsable page, so those two are declared as manual
downloads: the fetcher prints what to get and where to put it, then verifies.
"""
import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Optional

RAW_DIR = "data/raw"

# eMolecules keeps only the last few monthly snapshots online; the 2023-12-01
# release the paper used is no longer served.
EMOLECULES_SNAPSHOT = os.environ.get("MEEA_EMOLECULES_SNAPSHOT", "2026-11-01")


@dataclass
class Source:
    name: str
    dest: str                      # path under RAW_DIR
    url: Optional[str] = None      # None => manual download
    page: Optional[str] = None     # where a human gets it
    note: str = ""
    produces: List[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        return os.path.join(RAW_DIR, self.dest)


SOURCES = [
    Source(
        name="retro_star",
        dest="retro_data.zip",
        page="https://www.dropbox.com/s/ar9cupb18hv96gj/retro_data.zip?dl=0",
        note=(
            "Retro* data package, linked from https://github.com/binghong-ml/retro_star. "
            "Dropbox refuses scripted downloads, so fetch it in a browser and save it "
            "as data/raw/retro_data.zip. Unzipped it gives dataset/, one_step_model/ "
            "and saved_models/."
        ),
        produces=[
            "USPTO reactions + the official train/val/test split",
            "building block list (origin_dict.csv)",
            "test benchmark routes (routes_possible_test_hard.pkl)",
            "the one-step model MEEA* initialises from",
        ],
    ),
    Source(
        name="emolecules",
        dest=f"emolecules_{EMOLECULES_SNAPSHOT}.smi.gz",
        url=f"https://downloads.emolecules.com/free/{EMOLECULES_SNAPSHOT}/version.smi.gz",
        page=f"https://downloads.emolecules.com/free/{EMOLECULES_SNAPSHOT}/",
        note=(
            "Building blocks, ~350 MB gzipped. Lines are 'SMILES VERSION_ID PARENT_ID'. "
            "The paper cites the 2023-12-01 snapshot, but eMolecules only serves the "
            "last few monthly snapshots and that one is gone (404), so this defaults to "
            f"{EMOLECULES_SNAPSHOT}. Set MEEA_EMOLECULES_SNAPSHOT to pick another; "
            "https://downloads.emolecules.com/free/ lists what is available."
        ),
        produces=["prepare_data/origin_dict.csv"],
    ),
    Source(
        name="mol_props",
        dest="mol_props",
        page="https://weilab.math.msu.edu/DataLibrary/2D/",
        note=(
            "The nine molecular property datasets used as extra test sets. Download "
            "each one from the page and drop the files in data/raw/mol_props/ "
            "(csv or sdf, one file per dataset, named after it: logS, logP, "
            "LD50, Ames, BBBP, DPP4, BACE, SVS, ClinTox)."
        ),
        produces=["test_dataset/<name>.pkl"],
    ),
]

BY_NAME = {s.name: s for s in SOURCES}


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def describe(source: Source) -> str:
    lines = [f"[{source.name}] -> {source.path}", f"  {source.note}"]
    if source.url:
        lines.append(f"  url:  {source.url}")
    if source.page:
        lines.append(f"  page: {source.page}")
    for p in source.produces:
        lines.append(f"  gives: {p}")
    return "\n".join(lines)
