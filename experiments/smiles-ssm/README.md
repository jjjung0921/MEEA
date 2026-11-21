# SMILES 토큰 인코더 비교 실행 절차

계획은 [plan.md](plan.md)(v1, 승인). 구현은 `run.py` 하나와 `policyNet_variants.py`의
`SmilesTransformerPolicyNet(pool="mean")` · `SmilesBiSSMPolicyNet`이다.

## 실행

```bash
# 1) 데이터 지문과 검증 (출력 차원, 패딩 불변성, 유한 손실·gradient)
python experiments/smiles-ssm/run.py --verify --device cuda:0

# 2) 학습 — 같은 시드와 같은 배치 순서로 두 모델을 각각
python experiments/smiles-ssm/run.py --model transformer --device cuda:0 --epochs 30
python experiments/smiles-ssm/run.py --model ssm         --device cuda:0 --epochs 30

# 3) 마지막 epoch 기준 짝지은 비교
python experiments/smiles-ssm/run.py --compare
```

기본 출력 폴더는 `experiments/smiles-ssm/out/`이며 실행 설정·체크포인트·어휘표·항목별
정답 순위(`*_val_gold_rank.npy`, `val_gold_ranks.jsonl`)·요약(`summary.json`)이 쌓인다.
`--verify`가 실패하면 점수 비교를 보류한다.

## 이 저장소의 맥 환경에서 확인한 것

- 분할 무결성: 학습 35,634 / 검증 8,903, 파일 해시와 규칙 ID·표본 순서는 `out/dataset.json`.
- `max_len 160`에서 잘린 입력 0건 (학습 최대 153토큰, 검증 최대 122토큰).
- 검증셋 8,903건 중 1,882건은 정답 규칙이 학습 분할에 없다. 두 모델 공통의 상한이다.
- 파라미터 수: Transformer 5,827,655 / 양방향 SSM 5,828,679.
- 패딩 불변성 최대 오차 0.0, 손실·gradient 유한 — 두 모델 모두 통과.
- MPS는 이 환경의 torch 1.13.1에서 두 모델 모두 `MPSNDArray` assert로 죽는다. 맥에서는
  `--device cpu`만 쓸 수 있고, 측정값은 batch 256 기준 Transformer 약 1.66초/배치
  (약 244초/epoch), SSM 약 3.0초/배치(약 455초/epoch)로 30 epoch 두 모델에 약 6시간이다.
  그래서 학습은 데스크탑 GPU에서 수행한다.
