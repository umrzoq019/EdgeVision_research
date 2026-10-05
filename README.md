# EdgeVision

## Accuracy vs. Efficiency in Small CNNs

EdgeVision is a reproducible CPU-focused study of the accuracy/efficiency trade-off in deliberately small convolutional neural networks on CIFAR-10.

### Research question

> How much classification accuracy can be retained as a CNN is progressively reduced in parameter count and computational cost, and can compression techniques recover part of the lost accuracy without sacrificing the efficiency advantage?

## Current status — 5 October 2026

The core project is **research-complete enough to package and present**, but the repository should distinguish verified results from provisional ones.

### Current multi-seed result

| Model | Test accuracy | Parameters | Size | CPU latency |
|---|---:|---:|---:|---:|
| SmallCNNv2 | **61.83% ± 1.47** | 75,946 | 0.298 MB | 1.361 ms |
| TinyCNNv2 | **40.03% ± 1.53** | 4,218 | 0.0215 MB | 0.668 ms |
| NanoCNNv2 | **35.20% ± 0.78** | 1,250 | 0.0097 MB | 0.316 ms |
| Nano + distillation | **35.73% ± 1.27** | 1,250 | 0.0099 MB | 0.322 ms |
| Nano + INT8 | **PROVISIONAL / DO NOT CITE YET** | 1,250 expected | — | — |

The recorded Nano result removes **98.35% of the parameters** and **96.73% of the model size** relative to SmallCNNv2 while retaining 35.20% accuracy.

### Important quantization audit

An earlier standalone INT8 script defined a Nano architecture that did **not exactly match** the canonical `NanoCNNv2` in `train_v2.py`. Therefore its 34.10% / 0.008512 MB / 0.5617 ms result is retained as historical/provisional evidence but must not be presented as the final INT8 result.

The corrected quantization experiment must load the exact trained `NanoCNNv2` checkpoint before the paper is frozen.

## Repository

```text
EdgeVision_research/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── train_v2.py
│   ├── benchmark_repeated.py
│   ├── distill_v2.py
│   ├── distill_ablation.py
│   ├── quantize_dynamic_corrected.py
│   ├── validate_artifacts.py
│   └── plot_results.py
├── docs/
│   ├── REPRODUCE.md
│   ├── RESEARCH_STATUS.md
│   └── RESULT_PROVENANCE.md
├── results/
│   ├── verified/
│   └── provisional/
├── figures/
├── paper/
│   └── outline.md
└── .github/workflows/python-check.yml
```

## Quick start

```bash
python -m pip install -r requirements.txt
python -m py_compile src/*.py
```

Then follow [`docs/REPRODUCE.md`](docs/REPRODUCE.md) in order.

## What makes the project credible

- explicit model scaling rather than a single model;
- repeated CPU benchmarking;
- three-seed robustness for the main comparison;
- knowledge-distillation ablation;
- transparent negative/provisional results;
- reproducible scripts and documented experimental settings.

## Do not overclaim

This is an empirical CIFAR-10 study of custom CNNs. It does **not** establish that NanoCNN is generally optimal for edge deployment, that INT8 is universally beneficial, or that the measured latency transfers unchanged to every CPU/device.
