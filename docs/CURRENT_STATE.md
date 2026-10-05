# Current Project State

_Checkpoint covering the 3-seed baselines, 3-seed distillation, distillation ablation and first INT8 run._

## Completed

| # | Item | Evidence |
|---|---|---|
| 1 | Three small CNNs (Small / Tiny / Nano) trained on a 9,000-image CIFAR-10 subset, **3 seeds** (42, 123, 456) | `results/multi_seed/` |
| 2 | Sequential single-thread CPU latency benchmark (7 rounds × 500 runs) | `results/v2/latency_repeated.json` (legacy, see known issues) |
| 3 | SmallCNNv2 → NanoCNNv2 distillation: first run T=4, α=0.5, seed 42 (33.2 %) | `results/distillation/main.json` |
| 4 | Distillation ablation, 3 settings, seed 42 only | `results/distillation/ablation_metrics.json` |
| 5 | Distillation with the ablation's candidate (T=2, α=0.5) over **3 seeds** | `results/distillation/multi_seed/` |
| 6 | INT8 post-training static quantization of NanoCNNv2 (seed 42 only) | `results/quantization/` |
| 7 | Tables, paired statistics and figures regenerated from raw results | `results/analysis/`, `figures/` |

## Key numbers (3 seeds, 1,000-image test subset)

| Model | Test accuracy | Params |
|---|---:|---:|
| SmallCNNv2 | 61.83 ± 1.47 % | 75,946 |
| TinyCNNv2 | 40.03 ± 1.53 % | 4,218 |
| NanoCNNv2 | 35.20 ± 0.78 % | 1,250 |
| NanoCNNv2 + KD (T=2, α=0.5) | 35.73 ± 1.27 % | 1,250 |
| NanoCNNv2 INT8 (seed 42) | 34.10 % (FP32 same seed: 35.70 %) | 1,250 |

Distillation gain vs independent Nano: +0.53 pp (paired, 95 % CI −4.6 … +5.6, p ≈ 0.70) → no evidence of improvement.

## Not yet completed

- Latency re-measured with an interleaved protocol (script ready: `src/benchmark_interleaved.py`)
- INT8 re-run with interleaved timing, more seeds, Tiny/Small, QAT / other runtimes (script ready: `src/quantize_ptq.py`)
- Full 10,000-image test evaluation and paired tests (script ready: `src/eval_checkpoints.py`)
- More seeds (≥ 5, ideally 10) for baselines and distillation
- Distillation re-done with validation-based selection and controls
- Comparison with an established lightweight architecture (e.g. MobileNetV3-Small)
- Final Pareto analysis on trustworthy latency numbers

## Things to know before extending the work

- Do not overwrite `results/` when re-running; write to `*_rerun` directories (git-ignored) and promote deliberately.
- `*.pt` checkpoints live on Drive, not in git. The distillation scripts need the teacher checkpoints at `results/multi_seed/seed_<seed>/SmallCNNv2.pt`.
- Several earlier documents contained numbers that were misleading; corrections are listed in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
