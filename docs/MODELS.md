# EdgeVision v2 Models

All three baseline architectures are defined in `src/train_v2.py` and are trained/evaluated by the same pipeline.

| Model | Channels | Parameters |
|---|---|---:|
| SmallCNNv2 | 32 → 64 → 96 | 75,946 |
| TinyCNNv2 | 16 → 24 | 4,218 |
| NanoCNNv2 | 8 → 12 | 1,250 |

Each model uses 3×3 convolutions, BatchNorm, ReLU, max pooling and adaptive average pooling. The classifier maps the final channel representation to 10 CIFAR-10 classes.

NanoCNNv2 is also used as the student in the knowledge-distillation experiments, with SmallCNNv2 as teacher.
