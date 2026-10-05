# Data provenance

Which numbers are original machine output and which were reconstructed. When in doubt, prefer **raw** files over **derived** ones.

| File | Status | Notes |
|---|---|---|
| `results/multi_seed/raw_results.json`, `seed_*/metrics.json` | **Raw** (machine output of `multi_seed.py` / `train_v2.py`) | Includes per-epoch history. Latency fields are single 100-run timings taken right after training (noisy). |
| `results/multi_seed/summary.json` | Derived by `multi_seed.py` | Matches raw files. |
| `results/distillation/multi_seed/seed_*/metrics.json` | **Re-entered** from the original output | Transcribed from `distill_v2.py` metrics; the recomputed mean/SD (35.733 ± 1.270 %, latency 0.3217 ms) match the original summary exactly. |
| `results/distillation/multi_seed/summary.json` | Recomputed from the three files above | Same values as the original summary. |
| `results/distillation/ablation_metrics.json` | **Raw** (machine output of `distill_ablation.py`) | Authoritative ablation. Latency values unreliable. |
| `results/distillation/ablation.json` | **Reconstructed** from console output | Legacy; superseded by `ablation_metrics.json`. Latencies differ from the machine file. |
| `results/distillation/main.json` | Preserved from the first Colab run | T=4, α=0.5, seed 42, 33.2 %. |
| `results/v2/metrics.json` | Preserved from the first Colab run | Seed 42; accuracies equal `results/multi_seed/seed_42/metrics.json`, latencies differ (re-measured). |
| `results/v2/latency_repeated.json` | **Transcribed** from console output | Sequential benchmark, per-round values not kept; "distilled" entry is a different checkpoint from the 3-seed one. |
| `results/quantization/metrics.json`, `latency_repeated.json` | Original Colab output (per-round latencies kept) | Calibration source and original code not preserved. |
| `results/quantization/summary.txt`, `quantization_analysis.json` | Original derived summaries | Paths inside refer to the Colab Drive. |
| `results/analysis/*`, `figures/*` | **Generated** by `src/final_analysis.py` | Replaces the earlier CSVs/report; regenerate rather than edit. |

Not in the repository: model checkpoints (`.pt`), the original Colab notebook (export it), the original quantization code, hardware/library versions of the original runs.

Rule going forward: new runs go to new directories; `results/` entries above are never edited by hand.
