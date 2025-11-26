"""
TDC USPTO → atom-mapped → 템플릿 추출 → policyNet 데이터 생성

파이프라인:
1) TDC RetroSyn(name="USPTO") 로드 (input: 제품, output: 반응물).
2) RXNMapper로 forward 반응 `reactants>>product`에 atom map 부여.
3) rdchiral.template_extractor로 템플릿 추출.
4) policyNet 포맷 pkl 저장:
   - prepare_data/uspto_template.pkl.gz
   - prepare_data/policyTrain.pkl.gz (train+valid)
   - prepare_data/policyTest.pkl.gz (test)

주의:
- RXNMapper 실행 시간이 길 수 있습니다. --max-samples로 디버그 가능.
"""
import argparse
from collections import Counter
import gzip
import logging
import multiprocessing as mp
import os
import pickle
import signal
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from rxnmapper import RXNMapper
import rdchiral.template_extractor as te
from tdc.generation import RetroSyn


logger = logging.getLogger(__name__)


def pack_morgan_fp(smiles: str, fp_dim: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES 파싱 실패: {smiles}")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_dim, includeChirality=True)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(fp.GetNumBits(), dtype=np.uint8)
    arr[list(fp.GetOnBits())] = 1
    return np.packbits(arr)


def map_reactions(reactions: List[str], batch_size: int = 256) -> List[str]:
    """rxnmapper로 atom map 부여 (유효하지 않은 반응은 스킵)."""
    mapper = RXNMapper()
    mapped = []
    for i in tqdm(range(0, len(reactions), batch_size), desc="rxnmapper"):
        batch = reactions[i : i + batch_size]
        try:
            res = mapper.get_attention_guided_atom_maps(batch)
        except Exception:
            # batch에 문제가 있으면 건너뜀
            continue
        mapped.extend([r.get("mapped_rxn") for r in res if r.get("mapped_rxn")])
    return mapped


def extract_template_with_timeout(rxn: Dict, timeout_sec: float):
    """템플릿 추출을 시간 제한과 함께 실행."""
    if not timeout_sec or timeout_sec <= 0 or not hasattr(signal, "SIGALRM"):
        return te.extract_from_reaction(rxn)

    def _raise_timeout(signum, frame):
        raise TimeoutError("template extraction timeout")

    prev_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)
        return te.extract_from_reaction(rxn)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev_handler)


def build_dataset_from_mapped(
    mapped_rxns: List[str],
    fp_dim: int,
    template_list: List[str] = None,
    template_to_idx: Dict[str, int] = None,
    timeout_sec: float = 0,
    desc: str = "template extract",
) -> Tuple[List[np.ndarray], List[int], List[str], Dict[str, int], Counter]:
    """템플릿 추출 + fingerprint 구성. 타임아웃 시 건너뛰고 통계 수집."""
    if template_list is None:
        template_list = []
    if template_to_idx is None:
        template_to_idx = {tpl: i for i, tpl in enumerate(template_list)}

    stats = Counter({"total": 0, "success": 0, "failed": 0, "timeout": 0})
    fps: List[np.ndarray] = []
    labels: List[int] = []

    for rxn_str in tqdm(mapped_rxns, desc=desc):
        if ">>" not in rxn_str:
            stats["failed"] += 1
            continue
        stats["total"] += 1
        reactants, products = rxn_str.split(">>")
        rxn = {"_id": stats["total"], "reactants": reactants, "products": products, "spectators": ""}
        try:
            tpl = extract_template_with_timeout(rxn, timeout_sec=timeout_sec)
        except TimeoutError:
            stats["timeout"] += 1
            stats["failed"] += 1
            logger.warning(
                "Template extraction timed out after %.2fs for rxn %d: %s",
                timeout_sec,
                stats["total"],
                rxn_str[:200],
            )
            continue
        except Exception:
            stats["failed"] += 1
            continue

        if not isinstance(tpl, dict):
            stats["failed"] += 1
            continue
        rxn_smarts = tpl.get("reaction_smarts")
        if not rxn_smarts:
            stats["failed"] += 1
            continue
        try:
            fp = pack_morgan_fp(products, fp_dim=fp_dim)
        except Exception:
            stats["failed"] += 1
            continue

        if rxn_smarts not in template_to_idx:
            template_to_idx[rxn_smarts] = len(template_list)
            template_list.append(rxn_smarts)
        fps.append(fp)
        labels.append(template_to_idx[rxn_smarts])
        stats["success"] += 1
    return fps, labels, template_list, template_to_idx, stats


def dump_dataset(path: str, fps: List[np.ndarray], labels: List[int]):
    payload = {"smiles": np.array(fps, dtype=np.uint8), "template": np.array(labels, dtype=np.int64)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f)


def dump_mapped_list(path: str, mapped_rxns: List[str]):
    """매핑된 반응 문자열을 gzip+pkl로 저장."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(mapped_rxns, f)


def load_mapped_list(path: str) -> List[str]:
    """gzip+pkl로 저장된 매핑 결과를 로드."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"mapped 파일을 찾을 수 없습니다: {path}")
    with gzip.open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"매핑 파일 형식이 list가 아닙니다: {path}")
    return obj


def map_reactions_with_device(reactions: List[str], batch_size: int, gpu_id: str = None) -> List[str]:
    """옵션으로 CUDA_VISIBLE_DEVICES를 설정한 뒤 map_reactions 실행."""
    if gpu_id is None:
        return map_reactions(reactions, batch_size=batch_size)
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        return map_reactions(reactions, batch_size=batch_size)
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev


def map_worker(reactions: List[str], batch_size: int, gpu_id: str, out_dict, key: str):
    """multiprocessing용 worker."""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    out_dict[key] = map_reactions(reactions, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description="Map TDC USPTO and build policyNet data")
    parser.add_argument("--fp-dim", type=int, default=2048)
    parser.add_argument("--output-dir", default="prepare_data")
    parser.add_argument("--batch-size", type=int, default=256, help="rxnmapper batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="디버그용 샘플 제한")
    parser.add_argument(
        "--template-timeout",
        type=float,
        default=0,
        help="템플릿 추출 시간 제한(초). 0이면 제한 없음, 초과 시 해당 샘플은 실패로 카운트하고 건너뜀",
    )
    parser.add_argument(
        "--save-mapped",
        action="store_true",
        help="Train/Valid, Test 매핑 결과를 템플릿 추출 전에 gzip+pkl로 중간 저장",
    )
    parser.add_argument(
        "--load-mapped",
        default=None,
        help="중간 저장한 매핑 결과 폴더 경로 (mapped_train_valid.pkl.gz, mapped_test.pkl.gz 사용)",
    )
    parser.add_argument("--parallel-map", action="store_true", help="Train/valid와 test를 서로 다른 GPU에서 동시에 mapping")
    parser.add_argument("--train-gpu", default=None, help="Train/valid mapping에 사용할 GPU id (예: 0)")
    parser.add_argument("--test-gpu", default=None, help="Test mapping에 사용할 GPU id (예: 1)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    mapped_train: List[str] = []
    mapped_test: List[str] = []

    if args.load_mapped:
        train_path = os.path.join(args.load_mapped, "mapped_train_valid.pkl.gz")
        test_path = os.path.join(args.load_mapped, "mapped_test.pkl.gz")
        print(f"매핑 결과 로드 중... ({train_path}, {test_path})")
        mapped_train = load_mapped_list(train_path)
        mapped_test = load_mapped_list(test_path)
        print(f"불러온 매핑 수 - Train/Valid: {len(mapped_train)}, Test: {len(mapped_test)}")
    else:
        print("TDC USPTO 로드...")
        data = RetroSyn(name="USPTO")
        split = data.get_split()
        train_df = split["train"].reset_index(drop=True)
        valid_df = split["valid"].reset_index(drop=True)
        test_df = split["test"].reset_index(drop=True)
        train_all = train_df.append(valid_df, ignore_index=True)

        def prep_rxn_strings(df):
            """입력/출력이 비었거나 형식이 잘못된 반응을 건너뛴다."""
            if args.max_samples:
                df = df.iloc[: args.max_samples]
            rxns = []
            for _, row in df.iterrows():
                reactants = row["output"]
                product = row["input"]
                if not isinstance(reactants, str) or not isinstance(product, str):
                    continue
                if not reactants.strip() or not product.strip():
                    continue
                rxn_str = f"{reactants}>>{product}"
                if ">>" not in rxn_str or rxn_str.startswith(">>") or rxn_str.endswith(">>"):
                    continue
                rxns.append(rxn_str)
            return rxns

        train_rxns = prep_rxn_strings(train_all)
        test_rxns = prep_rxn_strings(test_df)

        print(f"Train/Valid 반응 수: {len(train_rxns)}, Test 반응 수: {len(test_rxns)}")

        # Map train/test (선택적으로 병렬)
        if args.parallel_map:
            print("Train/Valid & Test atom mapping (parallel)...")
            ctx = mp.get_context("spawn")
            manager = ctx.Manager()
            shared = manager.dict()
            procs = [
                ctx.Process(target=map_worker, args=(train_rxns, args.batch_size, args.train_gpu, shared, "train")),
                ctx.Process(target=map_worker, args=(test_rxns, args.batch_size, args.test_gpu, shared, "test")),
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError(f"{p.name} failed during mapping (exit code {p.exitcode})")
            mapped_train = shared.get("train", [])
            mapped_test = shared.get("test", [])
        else:
            print("Train/Valid atom mapping...")
            mapped_train = map_reactions_with_device(train_rxns, batch_size=args.batch_size, gpu_id=args.train_gpu)
            # Map and extract test
            print("Test atom mapping...")
            mapped_test = map_reactions_with_device(test_rxns, batch_size=args.batch_size, gpu_id=args.test_gpu)

        if args.save_mapped:
            dump_mapped_list(os.path.join(args.output_dir, "mapped_train_valid.pkl.gz"), mapped_train)
            dump_mapped_list(os.path.join(args.output_dir, "mapped_test.pkl.gz"), mapped_test)
            print("매핑 결과를 중간 저장했습니다 (template 추출 전에 재활용 가능).")

    print("Train/Valid 템플릿 추출...")
    train_fps, train_labels, template_list, template_to_idx, train_stats = build_dataset_from_mapped(
        mapped_train, fp_dim=args.fp_dim, timeout_sec=args.template_timeout
    )
    print(
        f"train 성공: {train_stats['success']}, 실패: {train_stats['failed']} (timeout {train_stats['timeout']}), 템플릿 수: {len(template_list)}"
    )

    print("Test 템플릿 추출...")
    test_fps, test_labels, template_list, template_to_idx, test_stats = build_dataset_from_mapped(
        mapped_test,
        fp_dim=args.fp_dim,
        template_list=template_list,
        template_to_idx=template_to_idx,
        timeout_sec=args.template_timeout,
        desc="template extract test",
    )
    print(
        f"test 성공: {test_stats['success']}, 실패: {test_stats['failed']} (timeout {test_stats['timeout']}), 템플릿 총합: {len(template_list)}"
    )

    # 저장
    out_dir = args.output_dir
    dump_dataset(os.path.join(out_dir, "policyTrain.pkl.gz"), train_fps, train_labels)
    dump_dataset(os.path.join(out_dir, "policyTest.pkl.gz"), test_fps, test_labels)
    with gzip.open(os.path.join(out_dir, "uspto_template.pkl.gz"), "wb") as f:
        pickle.dump(template_list, f)
    print("완료: policyTrain/Test 및 uspto_template를 저장했습니다.")


if __name__ == "__main__":
    main()
