"""
TDC USPTO → policyNet 데이터 변환 스크립트

입력:
  - PyTDC RetroSyn(name="USPTO") split (제품: input, 반응물: output)
  - rdchiral.template_extractor로 템플릿 추출 (atom mapping 필요)

출력 (기본: ./prepare_data):
  - uspto_template.pkl.gz      : 템플릿 SMARTS 리스트
  - policyTrain.pkl.gz         : {'smiles': packed_fps, 'template': labels} (train+valid)
  - policyTest.pkl.gz          : 동일 구조 (test)

메모:
  - PyTDC와 rdchiral이 사전에 설치돼 있어야 합니다.
  - 추출 실패/검증 실패하는 샘플은 건너뜁니다.
"""
import argparse
import gzip
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

import rdchiral.template_extractor as te
from tdc.generation import RetroSyn


def pack_morgan_fp(smiles: str, fp_dim: int = 2048) -> np.ndarray:
    """생성한 Morgan FP를 packbits로 압축."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES 파싱 실패: {smiles}")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_dim, includeChirality=True)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(fp.GetNumBits(), dtype=np.uint8)
    arr[list(fp.GetOnBits())] = 1
    return np.packbits(arr)


def extract_templates(rows, fp_dim: int, max_samples: int = None):
    """DataFrame rows에서 템플릿과 FP/레이블을 추출."""
    fps: List[np.ndarray] = []
    labels: List[int] = []
    template_list: List[str] = []
    template_to_idx: Dict[str, int] = {}

    iterable = rows.iterrows()
    if max_samples:
        iterable = list(rows.iterrows())[:max_samples]

    for idx, row in tqdm(iterable, total=(max_samples or len(rows)), desc="extract"):
        prod = row["input"]
        reactants = row["output"]
        rxn = {"_id": idx, "reactants": reactants, "products": prod, "spectators": ""}
        try:
            tpl = te.extract_from_reaction(rxn)
        except Exception:
            continue
        if not isinstance(tpl, dict):
            continue
        rxn_smarts = tpl.get("reaction_smarts")
        if not rxn_smarts:
            continue
        try:
            fp = pack_morgan_fp(prod, fp_dim=fp_dim)
        except Exception:
            continue

        if rxn_smarts not in template_to_idx:
            template_to_idx[rxn_smarts] = len(template_list)
            template_list.append(rxn_smarts)
        fps.append(fp)
        labels.append(template_to_idx[rxn_smarts])

    return fps, labels, template_list


def dump_dataset(path: str, fps: List[np.ndarray], labels: List[int]):
    payload = {"smiles": np.array(fps, dtype=np.uint8), "template": np.array(labels, dtype=np.int64)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f)


def main():
    parser = argparse.ArgumentParser(description="Build policyNet data from TDC USPTO")
    parser.add_argument("--fp-dim", type=int, default=2048)
    parser.add_argument("--output-dir", default="prepare_data")
    parser.add_argument("--max-samples", type=int, default=None, help="디버그용 샘플 제한")
    args = parser.parse_args()

    print("TDC USPTO 로드 중...")
    data = RetroSyn(name="USPTO")
    split = data.get_split()
    train_df = split["train"].reset_index(drop=True)
    valid_df = split["valid"].reset_index(drop=True)
    test_df = split["test"].reset_index(drop=True)

    # train = train + valid
    train_all = train_df.append(valid_df, ignore_index=True)

    print("Train/Valid에서 템플릿 추출...")
    train_fps, train_labels, template_list = extract_templates(
        train_all, fp_dim=args.fp_dim, max_samples=args.max_samples
    )
    print(f"train 유효 샘플: {len(train_labels)}, 템플릿 수: {len(template_list)}")

    # test split 템플릿 추출 (템플릿 사전에 없는 경우 새 템플릿 추가)
    print("Test에서 템플릿 추출...")
    test_fps: List[np.ndarray] = []
    test_labels: List[int] = []
    template_to_idx = {tpl: i for i, tpl in enumerate(template_list)}
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="extract test"):
        if args.max_samples and idx >= args.max_samples:
            break
        prod = row["input"]
        reactants = row["output"]
        rxn = {"_id": idx, "reactants": reactants, "products": prod, "spectators": ""}
        try:
            tpl = te.extract_from_reaction(rxn)
        except Exception:
            continue
        if not isinstance(tpl, dict):
            continue
        rxn_smarts = tpl.get("reaction_smarts")
        if not rxn_smarts:
            continue
        try:
            fp = pack_morgan_fp(prod, fp_dim=args.fp_dim)
        except Exception:
            continue

        if rxn_smarts not in template_to_idx:
            template_to_idx[rxn_smarts] = len(template_list)
            template_list.append(rxn_smarts)
        test_fps.append(fp)
        test_labels.append(template_to_idx[rxn_smarts])

    print(f"test 유효 샘플: {len(test_labels)}, 템플릿 총합: {len(template_list)}")

    # 저장
    out_dir = args.output_dir
    dump_dataset(os.path.join(out_dir, "policyTrain.pkl.gz"), train_fps, train_labels)
    dump_dataset(os.path.join(out_dir, "policyTest.pkl.gz"), test_fps, test_labels)
    with gzip.open(os.path.join(out_dir, "uspto_template.pkl.gz"), "wb") as f:
        pickle.dump(template_list, f)

    print("완료: policyTrain/Test 및 uspto_template를 저장했습니다.")


if __name__ == "__main__":
    main()
