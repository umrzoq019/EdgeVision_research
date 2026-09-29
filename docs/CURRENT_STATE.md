# Current Project State

Baseline V2:
- SmallCNNv2
- TinyCNNv2
- NanoCNNv2

Training:
- 30 epochs
- 10,000 training samples
- 1,000 validation samples
- 1,000 test samples
- batch size 128
- learning rate 0.001
- weight decay 0.0001
- seeds: 42, 123, 456

Multi-seed results:

SmallCNNv2:
61.83 ± 1.47% test accuracy
75,946 parameters
0.298 MB

TinyCNNv2:
40.03 ± 1.53% test accuracy
4,218 parameters
0.021 MB

NanoCNNv2:
35.20 ± 0.78% test accuracy
1,250 parameters
0.010 MB

Distillation:
Initial ablation completed.
Candidate configuration:
T=2, soft-target weighting=0.50

Next experiment:
Validate the selected distillation configuration across the
same three random seeds before evaluating quantization.

## Completed

1. Three small CNNs trained on CIFAR-10 using the v2 training pipeline.
2. Repeated single-thread CPU latency benchmark.
3. SmallCNNv2 → NanoCNNv2 knowledge distillation.
4. Three-setting distillation ablation.
5. v2 analysis script for accuracy/latency and validation-accuracy figures.

## Not yet completed

- INT8/post-training quantization
- established lightweight architecture comparison
- multi-seed statistical validation
- final Pareto analysis
