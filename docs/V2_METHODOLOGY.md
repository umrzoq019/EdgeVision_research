# EdgeVision v2 Methodology

## Objective

Measure the accuracy/efficiency trade-off among three deliberately small CNNs on CIFAR-10, test whether knowledge distillation can improve the smallest model without increasing its parameter count, and test whether INT8 post-training quantization reduces its size or latency.

## Data

CIFAR-10, 32×32 RGB, 10 classes. Deterministic subset protocol (seeded permutation):

- 10,000 training examples considered; 1,000 of them are the validation set, the remaining 9,000 are used for training
- first 1,000 images of the CIFAR-10 test set are used for final evaluation
- training augmentation: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`; evaluation: tensor conversion + CIFAR-10 normalization only
- the data split depends on the run seed (42, 123, 456), so seeds vary both initialization and the split

## Models

SmallCNNv2 (32→64→96), TinyCNNv2 (16→24), NanoCNNv2 (8→12): 3×3 conv + BatchNorm + ReLU blocks, max-pooling, adaptive average pooling, linear classifier. See [MODELS.md](MODELS.md).

## Training

AdamW, lr 0.001, weight decay 1e-4, batch 128, 30 epochs, cosine annealing, best-validation checkpoint restored before test evaluation. Seeds 42, 123, 456 (`src/multi_seed.py`). The original runs used the CPU runtime (`"device": "cpu"` in the raw output).

## Efficiency measurements

- **Parameters:** trainable parameters.
- **Size:** (a) serialized `state_dict` file size; (b) tensor payload bytes (parameters + BatchNorm buffers). For models this small the file size is mostly container overhead, so (b) is the meaningful number.
- **CPU latency, batch 1, one PyTorch thread.** Three generations of measurement exist:
  1. end-of-training timing in `train_v2.py` (100 runs, 20 warm-up) — noisy, kept for reference;
  2. sequential repeated benchmark (`benchmark_repeated.py`: 7 rounds × 500 runs, 50 warm-up) — each model timed in its own block;
  3. **recommended:** interleaved benchmark (`benchmark_interleaved.py`, `quantize_ptq.py`) — every round times all models in randomised order, per-round ratios and the environment (CPU model, torch version) are stored.

## Knowledge distillation

SmallCNNv2 (frozen, same seed) is the teacher and NanoCNNv2 the student. Loss = weight_hard · cross-entropy + weight_soft · T² · KL(softmax(teacher/T) ‖ softmax(student/T)).

- first run: T = 4, α = 0.5, seed 42 (`distill_v2.py`) — 33.2 % test accuracy;
- ablation (`distill_ablation.py`, seed 42): (T, soft weight) ∈ {(2, 0.25), (4, 0.25), (2, 0.50)};
- three-seed run with T = 2, α = 0.5 (`multi_seed_distill.py`).

The student architecture never changes. Note: in `distill_v2.py` `alpha` weights the hard-label loss, in `distill_ablation.py` `soft_weight` weights the KD loss; at 0.5 both coincide.

## Post-training INT8 quantization

PyTorch eager-mode static quantization: Conv-BN-ReLU fusion, `x86` engine, 1,000 calibration images, per-tensor default qconfig, test on the same 1,000 test images (`quantize_ptq.py`). The preserved result came from the original Colab code (not preserved); `quantize_ptq.py` re-implements the protocol and calibrates on the validation split.

## Statistics

Per-seed values are paired by seed. Reported: mean ± SD (n − 1), paired mean difference, two-sided paired t-test and 95 % t-interval (df = n − 1). With n = 3 these are weak; they exist to stop single-seed or ±1 pp differences from being over-read. A single test accuracy on 1,000 images has binomial SE ≈ 1.5 pp.

## Reproducibility notes

- Checkpoints are required to reproduce distillation (teachers) and quantization; they are on Drive, not in git.
- Exact reproduction of accuracy is not guaranteed across hardware/library versions; latency never is.
- The scripts' default output directories (`results_v2/`, `results_distill/`, …) are Colab-style and git-ignored; `multi_seed.py` writes to `results/multi_seed/` by default, so use `--output-root` for reruns.
