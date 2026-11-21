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

## 2026-11-21 GPU 학습 결과

시드 0, 배치 크기 256, 최대 입력 길이 160으로 두 모델을 30 epoch 학습했다.
학습 35,634건과 검증 8,903건, 토큰화·어휘, masked mean pooling, 분류 헤드를
공유했다. GPU에서 두 모델 모두 출력 차원, 패딩 불변성, 유한한 손실과 gradient
검사를 통과했다. 계획에 따라 마지막 epoch를 비교했다.

| 검증 지표 | Transformer | 양방향 SSM |
| --- | ---: | ---: |
| 파라미터 수 | 5,827,655 | 5,828,679 |
| Top-1 | 19.22% (1,711건) | 25.29% (2,252건) |
| Top-10 | 41.17% (3,665건) | 51.76% (4,608건) |
| Top-50 | 53.90% (4,799건) | 62.27% (5,544건) |
| 정답 순위 중앙값 | 30위 | 9위 |
| 학습 및 epoch별 검증 소요 시간 | 465.2초 | 2,773.0초 |

- **Top-50 차이 +8.37%p.** 같은 검증 반응을 짝지어 비교하면 SSM만 맞힌 경우가
  984건, Transformer만 맞힌 경우가 239건이다. 두 모델 모두 맞힌 경우는 4,560건이다.
- **학습에 등장한 규칙에서 개선.** 정답 규칙이 학습에 등장한 7,021건의 Top-50은
  68.35%에서 78.96%로 올랐다. 학습에 없던 정답 규칙 1,882건은 두 모델 모두 0건을 맞혔다.
- **학습 추이와 시간.** 검증 손실은 두 모델 모두 6 epoch에서 최저였다. Top-50 최고치는
  Transformer 19 epoch의 54.48%, SSM 30 epoch의 62.27%였다. SSM은 9 epoch
  (810.3초)에 Transformer의 최종 Top-50을 넘어섰으며, 30 epoch 기준 소요 시간은 5.96배였다.

기존 검증 분할과 단일 시드를 사용한 단일 단계 반응 규칙 분류 결과다. 시드 간 변동,
추론 속도, 다단계 경로 성공률은 측정 범위 밖이다.

[Transformer 로그](out/transformer.json) · [SSM 로그](out/ssm.json) ·
[데이터 지문](out/dataset.json) · [CUDA 검증](out/verification.json) · [공통 어휘](out/vocab.json) ·
[Transformer 정답 순위](out/transformer_val_gold_rank.npy) · [SSM 정답 순위](out/ssm_val_gold_rank.npy)

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
