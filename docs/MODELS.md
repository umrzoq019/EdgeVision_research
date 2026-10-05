# EdgeVision v2 Models

All three baseline architectures are defined in `src/train_v2.py` and trained and evaluated by the same pipeline. Parameter counts are checked in `tests/test_models.py`.

| Model | Conv channels | Parameters | Tensor payload* | Saved file |
|---|---|---:|---:|---:|
| SmallCNNv2 | 3 → 32 → 64 → 96 | 75,946 | 305,344 B | 0.298 MB |
| TinyCNNv2 | 3 → 16 → 24 | 4,218 | 17,208 B | 0.0215 MB |
| NanoCNNv2 | 3 → 8 → 12 | 1,250 | 5,176 B | 0.0097 MB |

\*Parameters and BatchNorm buffers in FP32. The saved `state_dict` file is larger because of container overhead, which dominates for NanoCNNv2.

Each block is 3×3 convolution (no bias) → BatchNorm → ReLU. SmallCNNv2 has three blocks with max-pooling after the first two; TinyCNNv2 and NanoCNNv2 have two blocks, each followed by max-pooling. Adaptive average pooling and one linear layer (→ 10 classes) finish each model.

NanoCNNv2 is also the student in the distillation experiments (teacher: SmallCNNv2) and the model used for INT8 quantization.
