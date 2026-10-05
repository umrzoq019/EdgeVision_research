# Known issues and corrections

Found while bringing the repository up to date. "Handled" means the documents and analysis were changed; "Open" means a new experiment is needed.

## Measurement

1. **Latency is noisy and not comparable across sessions — Open.**
   Nano's FP32 latency appears as 0.32 ms (end-of-training means), 0.37 ms (sequential benchmark median) and 0.53 ms (quantization session median). Small's end-of-training latency was 1.07, 1.90 and 1.11 ms for the three seeds. Colab CPU state, other tenants and the DataLoader workers still alive after training all plausibly contribute (not verified). Fix: `src/benchmark_interleaved.py` (round-robin, randomised order, per-round ratios, environment recorded).

2. **The "distilled Nano is slower" number (0.569 ms vs 0.375 ms) is an artifact — Handled.**
   `NanoCNNv2_distilled` and `NanoCNNv2` have identical architectures, so latency must be identical. The 0.569 ms value was most likely benchmarked from the *first* distilled checkpoint (the `benchmark_repeated.py` default path `results_distill/NanoCNNv2_distilled.pt` is the T=4, 33.2 % run; not verified), then merged by name with the 3-seed T=2 accuracy (35.73 %) in the old summary table, which produced "55.6 % latency reduction / 2.25× speed-up" for the distilled model. The analysis now attributes latency to the architecture.

3. **INT8 vs FP32 latency (0.94×) is inconclusive — Open.**
   In the preserved run the FP32 per-round timings fell from ≈ 0.54 ms to ≈ 0.35 ms (60 % drift) while INT8 stayed at 0.55–0.61 ms, and the two models appear to have been timed one after the other. The per-round INT8/FP32 ratio ranges 0.99–1.70. Re-run with `src/quantize_ptq.py`.

4. **File size is dominated by container overhead for tiny models — Handled (reporting).**
   NanoCNNv2 holds 1,250 parameters + BatchNorm buffers = 5,176 bytes of tensor data, but its saved `state_dict` is ≈ 9.7 KB. Size "reductions" between Nano variants or for INT8 (−13.4 %) mostly measure serialization. Tables now show both file size and tensor payload. Small differences between 0.0097, 0.0099, 0.0098 and 0.0104 MB across files are serialization differences, not model differences.

5. **Test set is 1,000 images — Open.** Binomial standard error ≈ 1.5 pp for accuracies between 35 % and 62 %; differences below ≈ 3 pp between single runs are not resolvable. Fix: `src/eval_checkpoints.py` on the full 10,000-image test set (SE ≈ 0.5 pp) plus McNemar tests.

## Methodology

6. **Ablation "best" setting was chosen on test accuracy — Handled (documented), Open (redo).**
   T=2 / soft 0.50 was selected as the candidate because it had the highest *test* accuracy (35.8 %). By *validation* accuracy it ranks last (36.8 % vs 37.6 % and 37.5 %). The three-seed run that followed reused the same 1,000 test images. With all gaps inside the noise this did not change conclusions, but selection must use validation data.

7. **The ablation is single-seed, and identical settings disagree — Handled (documented).**
   T=2 / soft 0.50 / seed 42 gave 35.8 % in `distill_ablation.py` and 35.0 % in the multi-seed `distill_v2.py` run. A likely cause (from reading the code, not tested): `distill_v2.py` constructs the teacher before the student after seeding, so the student starts from different random weights than in the ablation, where the RNG is re-seeded immediately before the student is built. Either way, 0.8 pp run-to-run variation is as large as the spread across ablation settings (1.2 pp).

8. **Quantization calibration source and code were not preserved — Open.**
   The result file records 1,000 calibration samples but not where they came from, and the Colab code is not in the repository. `src/quantize_ptq.py` is a re-implementation that calibrates on the validation split, never on test data. If the original calibrated on test images, that run was optimistic.

9. **Only seed 42 (the best-scoring Nano seed) was quantized — Open.**

## Documentation (all Handled in this update)

- `README.md` still described multi-seed, quantization and the ablation candidate as future work or omitted them.
- Old `CURRENT_STATE.md` was internally inconsistent (listed multi-seed results while also listing multi-seed validation as "not yet completed").
- `report.md` reported the distilled model's 55.6 % latency reduction (see issue 2), and its figure list referenced images that were not in the repository.
- `final_experiment_comparison.csv` placed INT8 latency (0.5617 ms, quantization session) next to FP32 Nano latency (0.316 ms, training-time measurement) as if comparable.
- `final_report.txt` listed all four models as "Pareto" using the artifactual latencies.
- Two different ablation files existed: `ablation.json` (reconstructed from console output; latencies 0.372/0.362/0.349 ms) and a machine-written `metrics.json` (latencies 0.692/0.331/0.338 ms, size 0.0104 MB). Test accuracies agree; the machine-written file is treated as authoritative and the other kept as legacy. Ablation latencies are unreliable in both.
- `readme.html` was the CIFAR-10 dataset's own redirect page, not part of the project; it is excluded.
- Many files shared the names `metrics.json`, `summary.json`, `latency_repeated.json`; they are now separated by directory (see DATA_PROVENANCE.md).
