# Next steps — detailed plan

Ordered so that each phase makes the next one trustworthy. Do Phase 0 first; Phases 1–2 fix the weakest parts of the current evidence and are cheap; Phases 3–6 are the real new science; Phases 7–8 finish the project.

Conventions: run from the repository root; send reruns to new directories (`*_rerun`, `results_v3/`), never into the preserved `results/` entries; after every phase, update `docs/CURRENT_STATE.md` and `docs/DATA_PROVENANCE.md`.

Time estimates come from the preserved training times on the CPU runtime used originally (per seed: baselines ≈ 18 min, distillation ≈ 7 min, one ablation setting ≈ 8 min). A Colab GPU runtime speeds up training only; latency must still be measured on CPU.

---

## Phase 0 — Publish this checkpoint and secure the artifacts (≈ 30 min)

1. **Replace the content of the GitHub repo with this folder.**
   ```bash
   git clone https://github.com/umrzoq019/EdgeVision_research.git
   cd EdgeVision_research
   git rm -r -q .                       # drop the outdated tracked files (keeps .git)
   cp -r /path/to/new/EdgeVision_research/. .
   git add -A
   git status                           # check: no data/, no *.pt
   git commit -m "Update to current results: 3-seed baselines/KD, INT8 PTQ, corrected docs, new scripts"
   git push origin main                 # branch may be 'master'
   ```
2. **Export the original Colab notebook** (File → Download → `.ipynb`) and commit it as `notebooks/original_colab.ipynb`. Keep `EdgeVision_v2_runner.ipynb` as the portable runner.
3. **Copy the checkpoints from Drive** (`results/multi_seed/seed_*/*.pt`, distillation and quantization `.pt` files) into the same relative paths locally (git-ignored). Verify with:
   ```bash
   python src/eval_checkpoints.py --checkpoints results/multi_seed/seed_42/NanoCNNv2.pt --out results/analysis/smoke_eval.json
   ```
   Expected: an accuracy in the mid-30s (the 1,000-image subset gave 35.7 %; the 10,000-image value can differ by about ±1.5 pp). Anything near 10 % means the wrong file or a loading problem. For archival, attach the `.pt` files to a GitHub Release (`git tag v2.0-results`, then Releases → attach) or use Git LFS.
4. **Add a `LICENSE`** (e.g. MIT) and optionally `CITATION.cff`.
5. **Record what the original runs used** if you can still find it (Colab runtime CPU model, `torch.__version__`) and add it to `docs/DATA_PROVENANCE.md`. Also write down where the 1,000 quantization calibration images came from (open issue 8 in KNOWN_ISSUES.md).

Done when: GitHub shows the new README, checkpoints are archived outside git, `pytest tests/` passes (`pip install -r requirements.txt && pytest -q`).

---

## Phase 1 — Fix latency measurement (≈ 1 h, no training)

Goal: replace every latency number with interleaved measurements that carry an environment record.

```bash
python src/benchmark_interleaved.py --checkpoint-root results/multi_seed --seed 42 \
    --rounds 30 --repeats 200 --warmup 100 --out results/latency/interleaved.json
```
Run it at least three times in separate sessions (new Colab runtime, or different times of day) and keep all three JSON files (`interleaved_session1.json`, …). Also run it once on a machine you control (laptop, Raspberry Pi) — the project is about edge devices.

Acceptance criteria:
- IQR/median < 10 % for every model; if not, close other processes / use a fixed machine.
- Small/Tiny/Nano median ratios agree across sessions within ≈ 15 %.
- `environment` is filled in (CPU model, torch version).

Then regenerate the analysis with the new numbers: `python src/final_analysis.py --latency results/latency/interleaved.json` (its `models` block has the same keys the script reads; the column names still say "sequential benchmark", so rename them in the script when you switch). Optionally add the three-session spread as error bars in `accuracy_vs_latency.png`. Update README/REPORT latency columns and delete the "indicative only" caveats only if the criteria above are met.

---

## Phase 2 — Fix statistics (≈ 3–5 h)

### 2a. Evaluate on the full 10,000-image test set (no retraining)

```bash
python src/eval_checkpoints.py \
  --checkpoints results/multi_seed/seed_{42,123,456}/{SmallCNNv2,TinyCNNv2,NanoCNNv2}.pt \
               results/distillation/multi_seed/seed_{42,123,456}/NanoCNNv2_distilled.pt \
  --out results/analysis/full_test_eval.json
```
(Bash brace expansion shown; list the files explicitly in other shells.) This gives accuracy with Wilson 95 % intervals (SE ≈ 0.5 pp) and McNemar p-values between every pair (e.g. Nano vs Nano-KD for the same seed). Add the 10k results as the primary accuracy column; keep the 1,000-image numbers as "original protocol".

### 2b. More seeds

At least 5, preferably 10 seeds for baselines and distillation. Run all seeds in one invocation into a new root (the script rewrites `raw_results.json` per invocation):
```bash
python src/multi_seed.py --seeds 42 123 456 789 1011 2024 7 99 314 2718 --output-root results/multi_seed_10
python src/multi_seed_distill.py --seeds 42 123 456 789 1011 2024 7 99 314 2718 \
    --baseline-root results/multi_seed_10 --output-root results/distillation/multi_seed_10
python src/final_analysis.py --baseline-dir results/multi_seed_10 --distill-dir results/distillation/multi_seed_10 \
    --out-dir results/analysis_10 --fig-dir figures_10
```
Extra cost for 7 new seeds ≈ 3 h of CPU time (much less on GPU). Decision rule to decide in advance: "distillation helps" only if the paired 95 % CI excludes 0 *and* the mean gain exceeds 1 pp.

---

## Phase 3 — A stronger protocol ("v3") (≈ 1–2 days)

The current protocol (9,000 training images, 1,000 test images, weak 62 % teacher) limits every conclusion. Define v3:

- full CIFAR-10 training set: 45,000 train / 5,000 validation; full 10,000 test set;
- 5+ seeds; GPU for training; CPU for latency;
- same architectures first (so v2 → v3 is a controlled change), optionally 60–100 epochs.

```bash
python src/multi_seed.py --seeds 42 123 456 789 1011 \
    --total-train 50000 --val-size 5000 --test-subset 10000 --epochs 60 \
    --output-root results_v3/multi_seed
```
Keep v3 results in a separate folder and document the protocol in a new `docs/V3_METHODOLOGY.md`. Expect much higher absolute accuracies and a stronger teacher — this is where distillation has a fair chance to show an effect.

---

## Phase 4 — Do distillation properly (≈ 1–2 days)

1. **Select on validation, report on test once.** Extend the grid, e.g. T ∈ {1, 2, 4, 8} × soft weight ∈ {0.25, 0.5, 0.75}:
   ```bash
   python src/distill_ablation.py --teacher results_v3/multi_seed/seed_42/SmallCNNv2.pt \
       --out-dir results_v3/ablation --settings "1:0.5,2:0.25,2:0.5,2:0.75,4:0.25,4:0.5,4:0.75,8:0.5" \
       --total-train 50000 --val-size 5000 --test-subset 10000
   ```
   Choose the best setting by `best_validation_accuracy_percent`, never by test accuracy. Then run `multi_seed_distill.py` with that setting for all seeds (add the flags `--total-train 50000 --val-size 5000 --test-subset 10000`).
2. **Add controls** that separate distillation from generic regularization: (a) label smoothing 0.1 without a teacher; (b) longer training for the independent Nano (same wall-clock as the distilled run). Without these, "KD helped" is not attributable to KD.
3. **Try variants only after 1–2:** logit-matching (MSE on logits), intermediate-feature hints through a 1×1 conv adapter, a medium teacher (Tiny as teacher for Nano), teacher-assistant chains (Small → Tiny → Nano).
4. Report paired seed differences with 95 % CIs and McNemar tests on the full test set.

---

## Phase 5 — Quantization, properly (≈ 1 day)

1. Re-run the existing experiment with interleaved timing and a recorded calibration source:
   ```bash
   python src/quantize_ptq.py --checkpoint results/multi_seed/seed_42/NanoCNNv2.pt --model NanoCNNv2 \
       --out-dir results/quantization_rerun --test-subset 10000
   ```
   Repeat for all seeds and for `TinyCNNv2` and `SmallCNNv2` (`--model`). Larger models are more likely to benefit.
2. Try `--engine fbgemm` and `--engine x86`; on ARM try `qnnpack`.
3. Report tensor payload (bytes) in addition to file size; for INT8 compute weight bytes analytically (1 byte per weight + scales/zero-points).
4. If eager-mode INT8 stays slower: test other runtimes — export to ONNX and run with ONNX Runtime INT8, or TFLite — and quantization-aware training (`torch.ao.quantization.prepare_qat`) if accuracy drops matter. Real speed-ups usually appear on ARM / dedicated INT8 paths, not on x86 eager PyTorch with tiny tensors.
5. Acceptance: INT8 vs FP32 compared with the per-round paired ratio (`comparison.per_round_ratio`) over ≥ 30 interleaved rounds, in ≥ 3 sessions.

---

## Phase 6 — Compare against established lightweight networks (≈ 1–2 days)

Add at least one standard model so the frontier has external reference points. Train from scratch with the same protocol (not ImageNet-pretrained), adapting the stem for 32×32 inputs. **Untested starter code** — verify parameter counts and forward shapes before use:

```python
import torchvision
def mobilenet_v3_small_cifar():
    m = torchvision.models.mobilenet_v3_small(weights=None, num_classes=10)
    m.features[0][0].stride = (1, 1)      # keep resolution on 32x32 inputs
    return m
def mobilenet_v2_035_cifar():
    m = torchvision.models.mobilenet_v2(weights=None, num_classes=10, width_mult=0.35)
    m.features[0][0].stride = (1, 1)
    return m
def shufflenet_v2_x05_cifar():
    return torchvision.models.shufflenet_v2_x0_5(weights=None, num_classes=10)
```
Steps: register them in `MODELS` in `src/train_v2.py` (add to the dict, and add the channel layout to `CHANNELS` in `final_analysis.py` or compute payload from the state dict); add MACs via `torch.utils.flop_counter.FlopCounterMode` (counts FLOPs on a `torch.randn(1,3,32,32)` forward) because parameters and CPU latency do not always agree; then train, benchmark with `benchmark_interleaved.py`, and add them to the tables. Where parameter counts differ by orders of magnitude, compare at matched cost as well (width multipliers).

---

## Phase 7 — Final Pareto study and write-up (≈ 1–2 days)

1. Axes: accuracy (full test set, mean ± CI) vs (a) interleaved CPU latency, (b) MACs, (c) tensor payload bytes. Mark non-dominated models using the CIs, not only the means.
2. Regenerate all tables/figures with `src/final_analysis.py`; extend it to read the v3, latency, full-test and baseline-model results. Drop the `non-dominated by mean` caveat only if the CIs separate.
3. Rewrite `docs/REPORT.md` around v3; keep v2 as "pilot protocol" with its limitations. Suggested structure: question → protocol → models → accuracy–cost frontier → what distillation and quantization did (with controls) → limitations → reproduction.
4. Tag the release (`git tag v3.0`), attach checkpoints and `results_v3/`.

---

## Phase 8 — Repository hygiene (ongoing, ≈ 2 h)

- CI: a GitHub Actions workflow running `pip install -r requirements.txt` and `pytest -q` on push (CPU only).
- Determinism: set `PYTHONHASHSEED`, keep `num_workers` and torch version in `settings`; add `torch.__version__` and CPU model to every metrics JSON (`latency_utils.environment_info()` already does this).
- Make results append-only: add a dated folder per run and a one-line entry to `docs/DATA_PROVENANCE.md` for each.
- README: keep the headline table generated from `results/analysis/` (copy numbers, don't retype).
- Add `LICENSE`, `CITATION.cff`, issue templates only if you plan outside contributors.

---

## Minimal path if time is short

Phase 0 → Phase 1 → Phase 2a → Phase 2b (5 seeds). That alone turns the current "suggestive" statements into defensible ones. Phases 3–6 are what make the project novel.
