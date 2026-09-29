# EdgeVision — Accuracy vs. Efficiency in Small CNNs

**EdgeVision** is a computer-vision research project studying the trade-off between classification accuracy and inference efficiency in deliberately small convolutional neural networks.

> **Research question:** How much accuracy can a vision model retain when its size and CPU inference cost are reduced?

The current checkpoint contains the completed **v2 baseline**, repeated CPU benchmarking, and an initial knowledge-distillation study. Quantization and comparisons with established lightweight networks are future work and are not presented as completed results.

## Project structure

```text
EdgeVision_research/
├── README.md
├── requirements.txt
├── docs/
│   ├── CURRENT_STATE.md
│   └── MODELS.md
├── notebooks/
│   └── EdgeVision_v2_runner.ipynb
├── src/
│   ├── train_v2.py
│   ├── analyze_v2.py
│   ├── distill_v2.py
│   ├── distill_ablation.py
│   └── benchmark_repeated.py
├── results/
│   ├── v2/
│   │   ├── metrics.json
│   │   └── latency_repeated.json
│   └── distillation/
│       ├── main.json
│       └── ablation.json
└── figures/
```

## Models

The model definitions live in `src/train_v2.py`; separate model files are not necessary because the architectures are small and are directly part of the reproducible training pipeline.

| Model | Parameters | Test accuracy | Model size | Reported CPU latency |
|---|---:|---:|---:|---:|
| SmallCNNv2 | 75,946 | 61.30% | 0.298 MB | 1.278 ms/image |
| TinyCNNv2 | 4,218 | 38.70% | 0.021 MB | 0.534 ms/image |
| NanoCNNv2 | 1,250 | 35.70% | 0.010 MB | 0.355 ms/image |

The three architectures use progressively smaller channel widths while keeping the general CNN structure comparable.

## Dataset and v2 protocol

- Dataset: CIFAR-10
- Input: 32×32 RGB images
- Classes: 10
- Training subset: 10,000 images
- Validation set: 1,000 images
- Test subset: 1,000 images
- Epochs: 30
- Batch size: 128
- AdamW learning rate: 0.001
- Weight decay: 1e-4
- Seed: 42
- CPU benchmark: 1 PyTorch thread

Training uses random crop and horizontal-flip augmentation. Evaluation uses normalized, non-augmented images.

## Main v2 result

The completed run produced the accuracy/efficiency measurements above. The large model retains substantially more test accuracy, while the smaller models use far fewer parameters and have lower measured CPU latency. This is the central empirical trade-off being investigated by EdgeVision.

The complete preserved summary is in `results/v2/metrics.json`.

## Repeated latency benchmark

`src/benchmark_repeated.py` measures CPU inference using 500 repetitions per round, 50 warm-up inferences, 7 rounds, and one PyTorch thread.

| Model | Median | Mean | Min | Max |
|---|---:|---:|---:|---:|
| SmallCNNv2 | 1.2824 ms | 1.2803 ms | 1.2273 ms | 1.3292 ms |
| TinyCNNv2 | 0.5468 ms | 0.5453 ms | 0.5210 ms | 0.5689 ms |
| NanoCNNv2 | 0.3749 ms | 0.4195 ms | 0.3546 ms | 0.6382 ms |
| NanoCNNv2 distilled | 0.5690 ms | 0.5651 ms | 0.5246 ms | 0.5962 ms |

The full result is preserved in `results/v2/latency_repeated.json`.

## Knowledge distillation

The first distillation experiment uses **SmallCNNv2 as teacher** and **NanoCNNv2 as student**. The main configuration used temperature 4.0 and alpha 0.5.

- Best validation accuracy: 34.5%
- Test accuracy: 33.2%
- Student parameters: 1,250
- Model size: ~0.00993 MB
- CPU latency: ~0.3696 ms/image

The independently trained NanoCNNv2 reference reached 35.7%, so the initial distillation run did not improve the student's test accuracy.

## Distillation ablation

Three configurations were tested:

| Method | Temperature | Soft-target weight | Test accuracy | Latency |
|---|---:|---:|---:|---:|
| T2_soft25 | 2 | 0.25 | 35.20% | 0.372 ms |
| T4_soft25 | 4 | 0.25 | 34.60% | 0.362 ms |
| T2_soft50 | 2 | 0.50 | 35.80% | 0.349 ms |

The best tested configuration was only 0.1 percentage point above the independent NanoCNNv2 result, so this experiment does not establish a meaningful distillation improvement.

## Analysis

`src/analyze_v2.py` reads `results_v2/metrics.json` and produces:

- `figures/accuracy_vs_latency.png`
- `figures/validation_accuracy.png`

The script expects the original Colab-style `results_v2/` directory. The preserved summaries in `results/v2/` are the archival record of the completed experiment.

## Running the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the v2 baselines:

```bash
python src/train_v2.py --epochs 30 --total-train 10000 --val-size 1000 --test-subset 1000
```

Run analysis from a Colab/local working directory containing the generated `results_v2/metrics.json`:

```bash
python src/analyze_v2.py
```

Run the main distillation experiment after the teacher checkpoint exists:

```bash
python src/distill_v2.py --epochs 30 --teacher results_v2/SmallCNNv2.pt
```

Run the ablation:

```bash
python src/distill_ablation.py
```

Run repeated latency benchmarking:

```bash
python src/benchmark_repeated.py
```

## Colab

The original working Colab notebook remains the execution environment for the project:

https://colab.research.google.com/drive/1bypBQJrqnrKD0ncEd4RWQjRP_QVjpxne?usp=sharing

The repository also contains `notebooks/EdgeVision_v2_runner.ipynb`, a portable runner that documents the main execution sequence.

**Important:** the runner notebook is not claimed to be an exported copy of the original Colab notebook. To preserve the exact original notebook, export/download the `.ipynb` from Colab and replace the runner with that file.

## What is not committed

- CIFAR-10 dataset files
- temporary Colab runtime files
- generated Python caches
- large model checkpoints unless deliberately added

The checkpoint files can be regenerated by the training/distillation scripts. If the exact trained `.pt` files are later downloaded from persistent Drive, they can be added under `artifacts/checkpoints/`.

## Limitations

The current evidence is a small experimental benchmark rather than a production deployment study. Results use a 1,000-image test subset and a single seed, and CPU latency depends on the hardware/software environment. The distillation ablation consists of single runs.

## Next research phase

After this repository checkpoint is preserved, the next planned experiments are:

1. post-training INT8 quantization;
2. FP32 vs INT8 accuracy, size and latency measurement;
3. quantization of TinyCNNv2/NanoCNNv2;
4. comparison with an established lightweight model such as MobileNetV3-Small;
5. repeated seeds for important configurations;
6. accuracy/latency/size Pareto analysis.
