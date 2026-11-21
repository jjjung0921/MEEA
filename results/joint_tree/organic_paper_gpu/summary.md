# MEEA*-PC policy comparison through BACE

The evaluation uses the public test lists from the authors' repository. Each policy has a budget of 500 policy calls or 600 seconds per target, 50 candidate templates per call, 100 lookahead simulations, and c_puct = 4. The stock and value checkpoint come from the authors' distribution. Policies use our 14,407-template benchmark checkpoints. Policy and value inference ran on an RTX 4060 Ti. RDKit template application and tree search ran on the CPU.

## Complete route success

| Dataset | Fingerprint MLP | Independent SSM | Joint SSM | Fingerprint Transformer | SMILES Transformer | SMILES + bond changes |
|---|---:|---:|---:|---:|---:|---:|
| logS | 155/407 (38.08%) | 156/407 (38.33%) | 155/407 (38.08%) | 120/407 (29.48%) | 119/407 (29.24%) | 117/407 (28.75%) |
| logP | 245/1,073 (22.83%) | 240/1,073 (22.37%) | 245/1,073 (22.83%) | 165/1,073 (15.38%) | 159/1,073 (14.82%) | 182/1,073 (16.96%) |
| LD50 | 248/872 (28.44%) | 245/872 (28.10%) | 241/872 (27.64%) | 179/872 (20.53%) | 174/872 (19.95%) | 196/872 (22.48%) |
| Ames | 230/1,129 (20.37%) | 229/1,129 (20.28%) | 229/1,129 (20.28%) | 139/1,129 (12.31%) | 174/1,129 (15.41%) | 182/1,129 (16.12%) |
| BBBP | 190/610 (31.15%) | 192/610 (31.48%) | 188/610 (30.82%) | 145/610 (23.77%) | 147/610 (24.10%) | 142/610 (23.28%) |
| DPP4 | 892/2,322 (38.42%) | 898/2,322 (38.67%) | 902/2,322 (38.85%) | 839/2,322 (36.13%) | 802/2,322 (34.54%) | 765/2,322 (32.95%) |
| BACE | 60/1,163 (5.16%) | 62/1,163 (5.33%) | 63/1,163 (5.42%) | 65/1,163 (5.59%) | 76/1,163 (6.53%) | 67/1,163 (5.76%) |
| **Micro-average** | 2,020/7,576 (26.66%) | 2,022/7,576 (26.69%) | 2,023/7,576 (26.70%) | 1,652/7,576 (21.81%) | 1,651/7,576 (21.79%) | 1,651/7,576 (21.79%) |

## Aggregate counts

| Policy | Complete routes of at least two steps | Timeouts |
|---|---:|---:|
| Fingerprint MLP | 1,977/7,576 (26.10%) | 6 |
| Independent SSM | 1,979/7,576 (26.12%) | 6 |
| Joint SSM | 1,980/7,576 (26.14%) | 6 |
| Fingerprint Transformer | 1,624/7,576 (21.44%) | 13 |
| SMILES Transformer | 1,619/7,576 (21.37%) | 20 |
| SMILES + bond changes | 1,619/7,576 (21.37%) | 23 |

The seven datasets contain **7,576 targets**, for **45,456** search runs. Value inference used sibling-state batching for 40,197 runs. Each affected target record carries a `batch_values` flag. The paired validation file records checks for route, outcome, and call-count agreement.

The run ends at BACE. SVS and ClinTox are outside this result. Earlier results on [Retro*-190](../retro190_paper/summary.md) use a separate target set.

The per-target records are stored in each dataset folder as `targets.jsonl.gz`.
