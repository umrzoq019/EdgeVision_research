# EdgeVision Research Log

## Research question
How much classification accuracy can a computer-vision model retain when its size and CPU inference cost are reduced for a resource-constrained system?

## Hypothesis
I expected smaller CNNs to use fewer parameters and have lower CPU latency than the larger baseline, but I also expected some loss in accuracy.

## Experimental setup
- Dataset: CIFAR-10
- Training examples: 5,000
- Test examples: 1,000
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Main comparison: 5 training epochs
- Models: SmallCNN, TinyCNN, NanoCNN
- Hardware for latency measurement: CPU in the execution environment

The first experiment used 2 epochs and compared SmallCNN with TinyCNN. A second experiment introduced NanoCNN. The main apples-to-apples comparison is Experiment 3, where all three models were trained for the same 5 epochs using the same dataset split and training settings.

## Run 1 — Baseline
- Date: September 17, 2026
- Training examples: 5,000
- Test examples: 1,000
- Epochs: 2
- Learning rate: 0.001
- SmallCNN accuracy: 29.40%
- TinyCNN accuracy: 25.90%
- SmallCNN parameters: 75,754
- TinyCNN parameters: 4,178
- SmallCNN model size: 0.293 MB
- TinyCNN model size: 0.019 MB
- SmallCNN CPU latency: 1.163 ms/image
- TinyCNN CPU latency: 0.560 ms/image

### Observation
TinyCNN was substantially smaller and faster, but it had lower accuracy in this short training run.

## Run 2 — Adding a smaller model
- Date: September 17, 2026
- Change: Added NanoCNN with fewer channels than TinyCNN.
- Reason for change: Test how far the architecture could be reduced before accuracy degraded substantially.
- Training examples: 5,000
- Test examples: 1,000
- Epochs: 2
- Learning rate: 0.001
- NanoCNN accuracy: 24.90%
- NanoCNN parameters: 1,230
- NanoCNN model size: 0.008 MB
- NanoCNN CPU latency: 0.298 ms/image

### Observation
NanoCNN used far fewer resources than TinyCNN, while its accuracy in the 2-epoch run was only 1.0 percentage point lower. This suggested that the smallest model still learned some useful signal, but the models needed a longer and consistent training run for a fairer comparison.

## Run 3 — Main comparison
- Date: September 17, 2026
- Change: Increased training from 2 to 5 epochs and evaluated all three architectures under the same settings.
- Reason for change: Determine whether the earlier accuracy differences were partly caused by limited training time and obtain a consistent three-model comparison.

| Model | Accuracy | Parameters | Model size | CPU latency |
|---|---:|---:|---:|---:|
| SmallCNN | 40.70% | 75,754 | 0.293 MB | 1.252 ms/image |
| TinyCNN | 30.20% | 4,178 | 0.019 MB | 0.475 ms/image |
| NanoCNN | 26.80% | 1,230 | 0.008 MB | 0.310 ms/image |

### Observations
Compared with SmallCNN:
- TinyCNN used about 94.5% fewer trainable parameters, was about 93.5% smaller by saved model size, and had about 62.1% lower measured CPU latency, while its accuracy was 10.5 percentage points lower.
- NanoCNN used about 98.4% fewer trainable parameters, was about 97.3% smaller by saved model size, and had about 75.2% lower measured CPU latency, while its accuracy was 13.9 percentage points lower.
- NanoCNN had only 1,230 trainable parameters and a measured size of about 0.008 MB, but its accuracy was also the lowest of the three models.

These results support the general direction of the hypothesis: reducing model capacity reduced resource usage, but the reduction came with an accuracy cost in this experiment.

## Final conclusion — draft
The experiment investigated the trade-off between accuracy and efficiency in three small convolutional neural networks trained on a 5,000-image subset of CIFAR-10. The main comparison used the same training settings for all three models over five epochs. SmallCNN achieved 40.70% test accuracy with 75,754 trainable parameters and a measured CPU latency of 1.252 ms per image. TinyCNN reduced the parameter count to 4,178 and latency to 0.475 ms, but accuracy fell to 30.20%. NanoCNN reduced the parameter count further to 1,230 and latency to 0.310 ms, while accuracy fell to 26.80%.

The main result is that making the model smaller produced substantial resource savings, but the smaller models did not retain the same classification accuracy. This suggests that model compression or architectural simplification involves a trade-off rather than a free improvement. An important limitation is that the experiment used only 5,000 training images and 1,000 test images for CIFAR-10 and only five training epochs. The CPU latency was also measured in a single execution environment, so it should not be treated as a universal hardware benchmark. A useful next step would be to test more training data, more epochs, and additional lightweight architectures on hardware closer to an actual robot.

## Questions to be able to answer in an interview
1. What is a convolution and why is it useful for images?
2. What is a trainable parameter in a neural network?
3. What loss function did you use, and why is it used during classification training?
4. What does backpropagation do?
5. Why might a smaller model be useful on a robot or other edge device?
6. Why does lower parameter count not automatically guarantee proportionally lower latency?
7. Why is Experiment 3 a fairer comparison than comparing the first two runs directly?
8. What limitations prevent these results from being treated as a general benchmark?
9. What would you test next if you had more compute or access to a small robot?
