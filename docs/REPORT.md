# EdgeVision — Accuracy vs. Efficiency in Small CNNs: current report

All numbers are generated from the preserved raw results by `src/final_analysis.py` (see `results/analysis/`). Measurement caveats are in [KNOWN_ISSUES.md](KNOWN_ISSUES.md); this report supersedes the earlier `report.md`.

## 1. Question and setup

How much CIFAR-10 accuracy is retained when a CNN is made drastically smaller and cheaper on CPU? Three CNNs (Small / Tiny / Nano), three seeds, 9,000 training images, 1,000 validation images, first 1,000 test images, 30 epochs, one CPU thread. Knowledge distillation (Small → Nano) and INT8 post-training quantization (Nano) were tested as ways to improve the smallest model. Protocol: [V2_METHODOLOGY.md](V2_METHODOLOGY.md).

## 2. Baselines

| Model | Test acc. (mean ± SD) | Val. acc. | Params | File | Payload | Median latency (sequential benchmark) | End-of-training latency (mean ± SD) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmallCNNv2 | 61.83 ± 1.47 % | 66.17 % | 75,946 | 0.2980 MB | 298 KiB | 1.282 ms | 1.36 ± 0.47 ms |
| TinyCNNv2 | 40.03 ± 1.53 % | 43.37 % | 4,218 | 0.0215 MB | 16.8 KiB | 0.547 ms | 0.67 ± 0.17 ms |
| NanoCNNv2 | 35.20 ± 0.78 % | 36.47 % | 1,250 | 0.0097 MB | 5.1 KiB | 0.375 ms | 0.316 ± 0.008 ms |

Relative to Small, Nano uses 98.35 % fewer parameters and 98.3 % less tensor payload, keeps 56.9 % of the accuracy (−26.6 pp), and is ≈ 3.4× faster by the sequential-benchmark medians (≈ 4× by the end-of-training means). Tiny keeps 64.7 % of the accuracy with 94.4 % fewer parameters (≈ 2.3× faster).

**Statistical reading (paired by seed, n = 3):**

| Comparison | Mean diff (pp) | 95 % CI | p |
|---|---:|---|---:|
| Small − Tiny | +21.8 | +15.6 … +28.0 | 0.004 |
| Small − Nano | +26.6 | +23.7 … +29.5 | 0.001 |
| Tiny − Nano | +4.8 | −0.9 … +10.5 | 0.067 |

The Small-vs-others gap is far larger than any noise source. Tiny vs Nano is plausible but not established with three seeds. Latency ordering Small > Tiny > Nano is robust; the exact ratios are not (latency of the same Nano model ranged 0.32–0.53 ms across sessions).

## 3. Knowledge distillation

| Setting | Seeds | Test accuracy |
|---|---:|---:|
| Independent NanoCNNv2 | 42, 123, 456 | 35.20 ± 0.78 % |
| Distilled, T=2, α=0.5 | 42, 123, 456 | 35.73 ± 1.27 % |
| Distilled, T=4, α=0.5 (first run) | 42 | 33.2 % |

Per-seed test accuracy (baseline → distilled): 35.7 → 35.0, 34.3 → 37.2, 35.6 → 35.0. Paired difference +0.53 pp (SD 2.05, 95 % CI −4.6 … +5.6, p ≈ 0.70). One seed (123) accounts for the whole mean gain; two of three seeds got worse. **There is no evidence that distillation helped under this setup.** The setup is unfavourable (the teacher reaches only ≈ 62 %, training uses 9,000 images), so this says little about distillation in general.

### Ablation (seed 42, single runs)

| Method | T | soft weight | Val. acc. | Test acc. | Rank (val / test) |
|---|---:|---:|---:|---:|---:|
| T2_soft25 | 2 | 0.25 | 37.6 % | 35.2 % | 1 / 2 |
| T4_soft25 | 4 | 0.25 | 37.5 % | 34.6 % | 2 / 3 |
| T2_soft50 | 2 | 0.50 | 36.8 % | 35.8 % | 3 / 1 |

Independent Nano with seed 42: 35.7 % test. The three settings span 1.2 pp; validation and test rankings are reversed; and the same nominal configuration as T2_soft50 produced 35.0 % in the multi-seed run (seed 42) versus 35.8 % here. The ablation therefore cannot rank settings. Its latency columns are unreliable (0.69 vs ≈ 0.33 ms for models with identical architecture).

## 4. INT8 post-training quantization (NanoCNNv2, seed 42 only)

| | FP32 | INT8 | Change |
|---|---:|---:|---:|
| Test accuracy (1,000 images) | 35.7 % | 34.1 % | −1.6 pp (binomial SE ≈ 1.5 pp per accuracy) |
| Validation accuracy | 38.9 % | 38.6 % | −0.3 pp |
| Saved file | 0.00983 MB | 0.00851 MB | −13.4 % |
| Median latency | 0.529 ms | 0.562 ms | 1.06× slower |

The accuracy change is within noise. The size change mostly reflects file container overhead (the FP32 tensor payload is only 5,176 bytes). The latency comparison is **inconclusive**: FP32 round timings fell from 0.54 to 0.35 ms inside the run (60 % drift) and INT8/FP32 per-round ratios range 0.99–1.70. That INT8 gives no speed-up is unsurprising for a 1,250-parameter model (quantize/dequantize overhead, tiny tensors), but it has not been measured properly. See `figures/quantization_latency_rounds.png`.

## 5. Interpretation

- EdgeVision's central hypothesis holds: removing capacity buys large reductions in parameters, payload and CPU time and costs a lot of accuracy; there is no free region in this model family.
- The three architectures trace an accuracy–cost frontier rather than a single winner: Small for accuracy, Tiny as a middle point, Nano for minimal footprint.
- Neither distillation (tested setup) nor INT8 PTQ (one seed) demonstrably moved Nano off that frontier.
- The Pareto table (`results/analysis/pareto_analysis.csv`) marks all architectures as non-dominated; the distilled Nano appears to dominate Nano only by a mean difference that is statistically indistinguishable from zero.

## 6. Limitations

- 3 seeds; 1,000-image test subset; 9,000 training images; single hardware/session per latency measurement; unrecorded environment for the original runs.
- Ablation and quantization are single-seed; ablation candidate was first picked by test accuracy.
- Results describe these architectures and this training budget, not optimal designs.

## 7. Figures

`figures/accuracy_vs_parameters.png`, `accuracy_vs_latency.png`, `accuracy_vs_size.png`, `seed_variability.png`, `distillation_ablation.png`, `quantization_latency_rounds.png`.
