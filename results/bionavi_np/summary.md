# BioNavi-NP evaluation with MEEA*-PC

MEEA*-PC found complete computational routes for **377 of 388 targets (97.16%)** in **6 min 13 s** on one NVIDIA GeForce RTX 4060 Ti with two workers.

| Metric | Result |
|---|---:|
| Benchmark targets | 388 |
| Targets with a complete route | 377 |
| Unsolved targets | 11 |
| Success rate | 97.16% |
| Mean route length among successes | 3.12 steps |
| Timeouts | 0 |
| Maximum policy calls observed | 46 |
| Maximum node expansions observed | 50 |

The MEEA* paper reports **97.68%** on BioNavi-NP. This run recovered two fewer targets, a difference of **0.52 percentage points**. Success means that the planner reached the designated building blocks within the search budget. [MEEA* paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10920677/)

## Evaluation setup

- **Targets and building blocks** — `test_dataset/natural.pkl` (388 targets) and the 40 unique SMILES in `building_blocks.txt` from the authors' MEEA* release.
- **Expansion and value** — The BioNavi-NP four-checkpoint OpenNMT ensemble (training steps 30,000, 50,000, 80,000, and 100,000), beam size 20, and the authors' MEEA*-PC `value_pc.pt` checkpoint. [BioNavi-NP repository](https://github.com/prokia/BioNavi-NP)
- **Search budget** — At most 100 policy calls and 50 node expansions per target. The runner used 100 lookahead simulations, c_puct = 4, and a 600-second wall-clock cap.
- **Scope** — This benchmark uses the dedicated BioNavi-NP expansion policy. The six policies trained for the organic-molecule benchmark were evaluated separately. [Organic evaluation through BACE](../joint_tree/organic_paper_gpu/summary.md)

The [execution conditions](conditions.json) record the upstream revision and hardware. The [machine-readable summary](summary.json) and [per-target records](targets.jsonl) contain the outcome, policy calls, node expansions, and elapsed time for each target. The evaluator is available in [scripts/evaluate_bionavi_np.py](../../scripts/evaluate_bionavi_np.py).
