# EdgeVision: Accuracy vs. Efficiency in Small CNNs

EdgeVision is a small computer-vision experiment investigating a practical question for resource-constrained AI systems:

> **How much classification accuracy can a vision model retain when its size and CPU inference cost are reduced?**

## What I tested

I implemented and compared three CNN architectures:

- **SmallCNN** — larger baseline
- **TinyCNN** — substantially reduced architecture
- **NanoCNN** — further reduced architecture

The models were trained on a 5,000-image subset of CIFAR-10 and evaluated on 1,000 test images. The main comparison used five epochs, the same learning rate, and the same batch size for all three models.

## Main results

| Model | Test accuracy | Parameters | Model size | CPU latency |
|---|---:|---:|---:|---:|
| SmallCNN | 40.70% | 75,754 | 0.293 MB | 1.252 ms/image |
| TinyCNN | 30.20% | 4,178 | 0.019 MB | 0.475 ms/image |
| NanoCNN | 26.80% | 1,230 | 0.008 MB | 0.310 ms/image |

The experiment showed a clear trade-off: reducing the architecture substantially lowered parameter count, saved model storage, and reduced measured CPU latency, but also reduced test accuracy.

## Why I did this

I am interested in computer vision for systems that cannot rely on large computing resources, such as robots or other edge devices. A model that performs well on a powerful computer may have different practical constraints when deployed on limited hardware. This experiment was a first attempt to measure that trade-off directly.

## Limitations

This is a small educational experiment, not a production benchmark. It used a limited subset of CIFAR-10, only five training epochs, and one CPU execution environment. The results therefore describe this experiment rather than proving that one architecture is universally better.

## Reproducibility

See `train.py`, `benchmark.py`, `experiment2.py`, `experiment3_fixed.py`, and `research_log.md` for the implementation and research notes. Run the code yourself and record any changes you make.
