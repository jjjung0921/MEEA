"""
Recover the SMILES string behind every row of policyTrain/policyTest.

policyTrain.pkl.gz stores packed Morgan fingerprints only. The products that
produced them live in data/uspto50k.tab. Re-fingerprinting every product and
matching on the packed bytes recovers the SMILES for each row, so the SMILES
dataset keeps the *exact* split and labels of the fingerprint dataset and the
two representations stay directly comparable.

Output: prepare_data/policySmilesTrain.pkl.gz / policySmilesTest.pkl.gz
        {'smiles': [str, ...], 'template': np.ndarray}
"""
import csv
import gzip
import pickle
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

FP_DIM = 2048


def packed_fp(smiles, gen):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(fp.GetNumBits(), dtype=np.uint8)
    arr[list(fp.GetOnBits())] = 1
    return np.packbits(arr).tobytes()


def main():
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=FP_DIM, includeChirality=True
    )

    fp2smiles = {}
    with open("data/uspto50k.tab") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            product = row["product"].strip().strip('"')
            key = packed_fp(product, gen)
            if key is not None:
                fp2smiles.setdefault(key, product)
    print(f"indexed {len(fp2smiles)} unique product fingerprints")

    for split in ("Train", "Test"):
        with gzip.open(f"prepare_data/policy{split}.pkl.gz", "rb") as f:
            data = pickle.load(f)
        fps, labels = data["smiles"], data["template"]

        smiles, keep = [], []
        for i in range(len(labels)):
            key = np.asarray(fps[i], dtype=np.uint8).tobytes()
            s = fp2smiles.get(key)
            if s is None:
                continue
            smiles.append(s)
            keep.append(i)

        # keep the packed fingerprint of the same rows so the fingerprint
        # models can train on exactly this subset
        out = {
            "smiles": smiles,
            "fp": np.asarray(fps)[keep],
            "template": np.asarray(labels)[keep],
        }
        path = f"prepare_data/policySmiles{split}.pkl.gz"
        with gzip.open(path, "wb") as f:
            pickle.dump(out, f)
        print(f"{split}: matched {len(keep)}/{len(labels)} -> {path}")
        if len(keep) < 0.9 * len(labels):
            print("WARNING: low match rate", file=sys.stderr)


if __name__ == "__main__":
    main()
