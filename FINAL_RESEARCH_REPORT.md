# Adversarial Robustness of Efficient Neural Networks
## Experimental Research Report

**Researcher:** Arman Md Mahfuj  
**Institution:** Suzhou University of Science and Technology  
**Research area:** Deep learning, adversarial robustness, and model compression

## Abstract

Deep neural networks can be vulnerable to adversarial perturbations, while model compression is commonly investigated as a way to reduce the computational cost of neural networks. This project studies the relationship between adversarial robustness and unstructured magnitude pruning using a small convolutional neural network on CIFAR-10. The experiments establish a baseline model, evaluate it against Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks, compare standard training with FGSM adversarial training, and investigate global magnitude pruning at 50% sparsity.

The baseline model achieved 55.53% clean test accuracy. Under PGD-50 evaluation with five random restarts, its robust accuracy was 2.42%. After pruning 50% of the eligible weights, clean accuracy decreased to 53.82% and robust accuracy decreased to 2.32%. Inference latency measurements did not show a consistent speed improvement. These preliminary findings suggest that, in the tested configuration, unstructured magnitude pruning alone did not improve adversarial robustness or consistently accelerate inference. Further experiments are needed to establish whether these observations generalize to other models, sparsity levels, and training procedures.

## 1. Introduction

Deep neural networks are used in image classification and many other machine-learning applications. However, small, carefully constructed changes to an input can cause a model to produce incorrect predictions. This behavior creates an important reliability concern for systems that use neural networks [1, 2].

At the same time, neural networks can require substantial computational resources. Model compression techniques, including pruning and quantization, aim to reduce model complexity or deployment costs. An important research question is whether compression affects a model's resistance to adversarial attacks.

This project investigates the relationship between model sparsity, clean accuracy, adversarial robustness, and inference latency. The initial experiments focus on global unstructured magnitude pruning, which removes weights with relatively small absolute values across selected layers.

The work is an experimental baseline for a master's research project rather than a complete evaluation of efficient neural-network robustness.

## 2. Research Questions

The experiments address four questions:

1. How vulnerable is the baseline CNN to FGSM and PGD attacks?
2. How does FGSM adversarial training affect clean accuracy and adversarial accuracy?
3. How does 50% global magnitude pruning affect clean accuracy and PGD robustness?
4. Does unstructured pruning produce a measurable inference-speed improvement on the available GPU?

## 3. Experimental Setup

### 3.1 Dataset and model

The main pruning experiments use CIFAR-10, which contains 50,000 training images and 10,000 test images across ten classes. Images are converted to tensors with pixel values in the range [0, 1].

The classifier is a small convolutional neural network consisting of three convolutional layers with ReLU activations, two max-pooling operations, adaptive average pooling, and a linear classification layer.

The model contains 94,538 parameters, including 94,304 eligible convolutional and linear weights for the pruning experiment.

### 3.2 Baseline training

The baseline model was trained for five epochs using the Adam optimizer with a learning rate of 0.001 and a batch size of 64. The recorded random seed was 42.

The short training duration and resulting accuracy should be considered when interpreting all subsequent results. The experiments measure robustness for this particular model and training configuration.

### 3.3 Adversarial attacks

Two gradient-based attacks were used:

- **FGSM:** A single-step attack that perturbs an input in the direction determined by the sign of the loss gradient.
- **PGD:** An iterative attack that repeatedly updates the input and projects the perturbation back into the allowed range. 

For the principal PGD-50 pruning comparison, the maximum perturbation was ε = 8/255, the step size was α = 2/255, and five random restarts were used. Perturbed pixel values were constrained to [1, 2].

### 3.4 Adversarial training

A separate model was trained using FGSM-generated adversarial examples with a training perturbation of ε = 8/255 for five epochs. Its results were compared with those of the standard baseline model.

### 3.5 Global magnitude pruning

Global unstructured magnitude pruning was applied to the weights of convolutional and linear layers. Biases were excluded. The 50% pruning configuration targeted half of the eligible weights, and the resulting zeros were made permanent before the pruned checkpoint was saved.

The baseline checkpoint was preserved  [3].

### 3.6 Inference latency

The baseline and pruned models were benchmarked on an NVIDIA GeForce GTX 1050 using CUDA. The benchmark used 20 warm-up forward passes and 100 timed forward passes for batch sizes of 1 and 64. Inputs were already on the GPU, and inference mode was enabled.

This procedure measures model forward-pass latency under the tested conditions. It does not include data loading or host-to-device transfer time.

## 4. Results

### 4.1 Baseline vulnerability

The baseline model achieved 55.53% clean test accuracy.

| Evaluation | Accuracy |
|---|---:|
| Clean images | 55.53% |
| FGSM, ε = 2/255 | 34.57% |
| FGSM, ε = 4/255 | 20.54% |
| FGSM, ε = 8/255 | 7.61% |
| PGD-10, ε = 8/255 | 3.46% in the model comparison |
| PGD-50, five random restarts, ε = 8/255 | 2.42% |

The results show a substantial decline in accuracy under adversarial perturbations. The PGD-50 result indicates that the baseline model was highly vulnerable under the specified evaluation protocol.

### 4.2 Adversarial training

| Evaluation | Standard model | FGSM adversarially trained model |
|---|---:|---:|
| Clean accuracy | 55.53% | 45.88% |
| FGSM, ε = 2/255 | 34.57% | 41.39% |
| FGSM, ε = 4/255 | 20.54% | 37.12% |
| FGSM, ε = 8/255 | 7.61% | 28.87% |
| PGD-10, ε = 8/255 | 3.46% | 27.68% |

In these experiments, FGSM adversarial training increased adversarial accuracy but reduced clean accuracy. The result illustrates a trade-off between clean performance and robustness for this training setup.

These figures are preliminary because the models were trained for only five epochs and the comparison does not establish performance across multiple independent training seeds.

### 4.3 Effect of 50% magnitude pruning

| Metric | Baseline | 50% pruned |
|---|---:|---:|
| Clean accuracy | 55.53% | 53.82% |
| PGD-50 accuracy, five restarts | 2.42% | 2.32% |
| Eligible weight sparsity | 0% | 50% |
| Checkpoint size | 381,668 bytes | 381,788 bytes |

Pruning reduced clean accuracy by 1.71 percentage points and PGD-50 accuracy by 0.10 percentage points. The measured robust accuracy therefore did not improve after pruning.

The pruned checkpoint was 120 bytes larger than the original checkpoint. This small difference does not indicate that pruning inherently increases model size; rather, it shows that zeroing weights in a dense checkpoint does not necessarily reduce storage.

### 4.4 Inference latency

| Batch size | Model | Latency per batch | Throughput |
|---:|---|---:|---:|
| 1 | Baseline | 0.8415 ms | 1,188.39 images/s |
| 1 | 50% pruned | 0.8272 ms | 1,208.96 images/s |
| 64 | Baseline | 5.9306 ms | 10,791.47 images/s |
| 64 | 50% pruned | 6.2054 ms | 10,313.65 images/s |

At batch size 1, the pruned model was approximately 1.7% faster. At batch size 64, it was approximately 4.6% slower.

Because the benchmark was run once for each configuration, these small differences may reflect timing variation or measurement noise. Repeated measurements are necessary before claiming a genuine performance difference.

The experiment does not demonstrate a consistent inference-speed benefit from unstructured magnitude pruning on this hardware and model.

### 4.5 Additional CIFAR-100 experiments

Additional scripts and result files in the project cover CIFAR-100 baseline and adversarial-training experiments. The recorded results show that FGSM adversarial training increased PGD-10 accuracy from 2.33% to 9.86%, while clean accuracy decreased from 24.61% to 17.84%.

These results provide additional preliminary evidence of a clean-accuracy versus adversarial-robustness trade-off. They should be presented separately from the CIFAR-10 pruning comparison because the dataset and experimental configuration differ.

## 5. Discussion

The experiments provide three main observations.

First, the baseline CNN was vulnerable to gradient-based adversarial attacks. Its accuracy decreased sharply as the attack strength increased, and iterative PGD produced lower accuracy than FGSM in the recorded comparisons.

Second, FGSM adversarial training improved adversarial accuracy in the tested configurations but reduced clean accuracy. This suggests that the training procedure changes the model's behavior under both clean and perturbed inputs. It does not, by itself, establish robustness against every attack or threat model.

Third, 50% global magnitude pruning did not improve PGD-50 accuracy in the tested model. It also did not consistently improve inference latency. The results indicate that sparsity alone should not be treated as evidence of improved robustness or computational efficiency.

These findings are limited to the selected architecture, training duration, dataset, pruning method, hardware, and attack settings. They do not establish that pruning is always ineffective or that pruning combined with adversarial training cannot be beneficial.

## 6. Limitations

The current experiments have several limitations:

1. **Limited training:** The models were trained for five epochs, and the baseline's clean accuracy was relatively low.
2. **Limited training seeds:** The main comparisons do not establish variation across multiple independently trained models.
3. **Limited pruning coverage:** The principal pruning comparison evaluates one sparsity level, 50%.
4. **Limited architecture coverage:** The main pruning study uses one small CNN rather than a range of architectures.
5. **Latency uncertainty:** The benchmark was performed once per configuration, without repeated trials or confidence intervals.
6. **Dense representation:** Unstructured zeros do not automatically reduce checkpoint storage or accelerate standard dense convolution operations.
7. **Attack coverage:** The results concern the specified attacks and parameters, not all possible adversarial threats.
8. **Limited generalization:** The results should not be generalized to other datasets, models, hardware, or deployment settings without additional evidence.

## 7. Future Work

The next research steps should prioritize the following:

1. Repeat latency measurements and report variability.
2. Train models for longer and evaluate whether the conclusions hold at higher clean accuracy.
3. Repeat the main comparisons using multiple training seeds.
4. Evaluate additional pruning levels, such as 30%, 50%, and 70%.
5. Compare pruning applied to standard training with pruning applied to adversarially trained models.
6. Evaluate whether adversarial fine-tuning after pruning improves the robustness–accuracy trade-off.
7. Investigate structured pruning or sparse inference implementations if the goal is to reduce real computational cost.
8. Report clean accuracy, robust accuracy, sparsity, checkpoint size, and latency as separate metrics.

## 8. Conclusion

This project established an experimental pipeline for evaluating clean accuracy, adversarial robustness, magnitude pruning, and inference latency in a small CNN.

The baseline model showed substantial vulnerability to FGSM and PGD attacks. FGSM adversarial training improved adversarial accuracy in the recorded experiments while reducing clean accuracy. Applying 50% global magnitude pruning to the baseline reduced clean accuracy from 55.53% to 53.82% and PGD-50 accuracy from 2.42% to 2.32%. The latency measurements showed a small improvement at batch size 1 but slower execution at batch size 64.

Consequently, the current results do not demonstrate that 50% unstructured magnitude pruning improves adversarial robustness or consistently accelerates inference. The experiments form a useful preliminary baseline for further study of robustness-aware compression, but more extensive and repeated evaluations are required before drawing general conclusions.

## Reproducibility and project artifacts

The project repository contains the training and evaluation scripts, model checkpoints, experiment CSV files, pruning code, and latency benchmark. The main pruning-related artifacts include:

- `prune_baseline.py`
- `compare_pruned_clean.py`
- `compare_pruned_pgd50.py`
- `benchmark_pruned_latency.py`
- `experiments/pruning_baseline_results.csv`
- `experiments/pruning_clean_comparison.csv`
- `experiments/pruned_pgd50_5restarts_comparison.csv`
- `experiments/pruning_latency_results.csv`
- `PRUNING_LATENCY_REPORT.md`

The results should be interpreted alongside the recorded configurations and limitations. Before formal submission, verify all figures against the corresponding CSV files and ensure the final report follows the supervisor's required format.
## References

[1] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and Harnessing Adversarial Examples,” *International Conference on Learning Representations (ICLR)*, 2015. https://arxiv.org/abs/1412.6572

[2] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards Deep Learning Models Resistant to Adversarial Attacks,” *International Conference on Learning Representations (ICLR)*, 2018. https://arxiv.org/abs/1706.06083

[3] S. Han, H. Mao, and W. J. Dally, “Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding,” *International Conference on Learning Representations (ICLR)*, 2016. https://arxiv.org/abs/1510.00149