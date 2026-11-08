# Adversarial Robustness of Efficient Neural Networks
## Experimental Research Report

**Researcher:** Arman Md Mahfuj  
**Institution:** Suzhou University of Science and Technology  
**Research area:** Deep learning, adversarial robustness, and model compression  
**Dataset:** CIFAR-10 (primary); CIFAR-100 (exploratory supporting experiments)

## Abstract

This project investigates how FGSM adversarial training and global unstructured magnitude pruning affect the clean accuracy and adversarial robustness of a CIFAR-adapted ResNet-18. The principal evaluation uses a methodologically controlled CIFAR-10 split of 45,000 training images, 5,000 validation images, and 10,000 held-out test images. The standard checkpoint was selected using validation accuracy rather than test-set performance. Its clean test accuracy was 88.06%. FGSM adversarial fine-tuning increased adversarial accuracy under the evaluated FGSM and PGD attacks, including PGD-10 accuracy from 0.00% to 33.49% and PGD-50 robust accuracy from 0.00% to 32.25%, while clean accuracy decreased to 68.93%.

Global L1 unstructured magnitude pruning at 30%, 50%, and 70% was evaluated separately for the standard and FGSM-trained models. Moderate pruning largely preserved performance, while 70% pruning caused a substantial clean-accuracy reduction for the FGSM-trained model. The tested dense-GPU benchmark did not show meaningful latency or checkpoint-size improvements from unstructured pruning. The findings demonstrate a clean-accuracy and adversarial-robustness trade-off for the tested FGSM training procedure, and show that unstructured sparsity alone did not provide a practical efficiency gain in the measured setup. All principal results are single-seed experiments and should be interpreted within the stated limitations.

## 1. Introduction

Deep neural networks can be vulnerable to adversarial examples: inputs modified by small, deliberately chosen perturbations that cause incorrect predictions. Adversarial robustness is therefore an important reliability concern for image-classification systems [1, 2].

Model compression methods such as pruning aim to reduce model complexity and deployment cost. However, removing weights does not automatically improve adversarial robustness or accelerate inference, particularly when sparse weights are still stored and processed using dense tensor operations. Research has consequently examined the relationship between compression and adversarial robustness [3, 4].

This project studies two connected questions: how FGSM adversarial training changes the robustness of a CIFAR-10 classifier, and how global unstructured magnitude pruning affects the clean and adversarial performance of standard and adversarially trained models. The principal study uses a CIFAR-adapted ResNet-18 and a held-out test set. Earlier SimpleCNN, CIFAR-100, and test-selected ResNet-18 experiments are retained as supporting or historical work rather than mixed into the principal comparison.

## 2. Research Questions

1. How vulnerable is a validation-selected CIFAR-adapted ResNet-18 to FGSM and PGD attacks?
2. How does FGSM adversarial fine-tuning change clean accuracy and adversarial accuracy under matched evaluations?
3. How do 30%, 50%, and 70% global unstructured magnitude-pruning levels affect standard and FGSM-trained models?
4. Does unstructured pruning produce measurable inference-latency or dense-checkpoint-size improvements on the available GPU?

## 3. Experimental Methodology

### 3.1 Dataset and data split

The principal experiments use CIFAR-10, which contains 50,000 training images and 10,000 test images across ten classes. The original training set was split into:

- 45,000 images for model training;
- 5,000 images for validation and checkpoint selection;
- 10,000 test images held out for final evaluation.

The split used random seed 42. Images were converted to tensors with pixel values in the range [0, 1]. The test set was not used to select the principal standard-model checkpoint.

CIFAR-100 experiments were conducted separately with SimpleCNN as exploratory supporting work. They were short, single-seed experiments and are not part of the principal ResNet-18 comparison.

### 3.2 Model architecture and standard training

The principal classifier is a CIFAR-adapted torchvision ResNet-18. Its ImageNet-oriented stem was modified to use a 3 × 3 first convolution with stride 1 and padding 1, and the initial max-pooling layer was replaced with an identity operation. The final fully connected layer outputs ten classes. The model contains 11,173,962 parameters.

The standard model was trained for 100 epochs using SGD, an initial learning rate of 0.1, batch size 64, and weight decay 0.0005. A learning-rate scheduler was used. The best checkpoint was selected by validation accuracy; the best recorded validation accuracy was 89.70% at epoch 54. The selected checkpoint achieved 88.06% clean accuracy on the held-out test set.

### 3.3 Adversarial attacks and evaluation metrics

The study used untargeted gradient-based attacks:

- **FGSM:** a single-step perturbation based on the sign of the input-loss gradient.
- **PGD:** an iterative attack that updates the input and projects the perturbation into the allowed L-infinity ball and valid pixel range.

The principal attack settings were:

| Attack | Maximum perturbation (ε) | Step size (α) | Steps | Random restarts |
|---|---:|---:|---:|---:|
| FGSM | 2/255, 4/255, 8/255 | Single step | 1 | Not applicable |
| PGD-10 | 8/255 | 2/255 | 10 | 1 random start |
| PGD-50 | 8/255 | 2/255 | 50 | 5 |

Inputs were constrained to the pixel range [0, 1]. The custom FGSM and PGD implementations were compared with the established `torchattacks` implementation for the tested CIFAR-10 ResNet-18 settings. This comparison supports the tested configurations; it does not validate every possible attack configuration.

The report distinguishes the following metrics:

- **Clean accuracy:** percentage of all test images classified correctly without perturbation.
- **Adversarial accuracy / robust accuracy:** percentage of all test images classified correctly under the specified attack.
- **Attack success rate (ASR):** percentage of initially correctly classified test images that are misclassified after the attack. The denominator is therefore the number of clean-correct samples, not the full test-set size.

For multi-restart PGD-50, cumulative ASR counts a clean-correct image as successfully attacked if at least one of the five restarts causes misclassification. Robust accuracy is calculated over the full test set.

### 3.4 FGSM adversarial fine-tuning

A separate ResNet-18 model was fine-tuned using FGSM-generated adversarial examples with a training perturbation of ε = 8/255. Fine-tuning used 45,000 training images and 5,000 validation images for 10 epochs, batch size 32, SGD with learning rate 0.001, momentum 0.9, and weight decay 0.0005. The learning-rate schedule used milestones at epochs 6 and 8 with a decay factor of 0.1. The best checkpoint was selected using validation performance.

The standard and FGSM-trained models were evaluated on the same held-out test set using the attack settings described above.

### 3.5 Global unstructured magnitude pruning

Global L1 unstructured magnitude pruning was applied at 30%, 50%, and 70% sparsity to the eligible weights of convolutional and linear layers. Pruning was evaluated separately for the validation-based standard model and the FGSM-trained model. Biases were excluded. The resulting pruning masks were preserved for the corresponding pruned-model evaluation and fine-tuning procedures.

Clean accuracy, adversarial accuracy, and PGD-10 ASR were considered together. ASR was not interpreted in isolation because a change in the number of initially correct samples can affect the conditional denominator.

### 3.6 Inference-efficiency benchmark

A preliminary benchmark compared a ResNet-18 baseline with a fine-tuned 70%-pruned model on an NVIDIA GeForce GTX 1050 using CUDA. The batch size was 64, with 50 warm-up iterations and 100 timed iterations in each of 10 measurement rounds. The reported latency is the median across rounds.

This benchmark measured forward-pass performance under the tested setup. It did not use sparse inference kernels or compressed sparse storage. Therefore, the results assess the practical effect of unstructured zero-valued weights in the tested dense execution path, not the theoretical potential of specialized sparse hardware or software.

## 4. Results

### 4.1 Validation-based standard ResNet-18

The validation-selected standard model achieved 89.70% validation accuracy and 88.06% clean test accuracy. Its adversarial accuracy declined substantially as attack strength increased.

| Evaluation | Adversarial accuracy |
|---|---:|
| Clean test images | 88.06% |
| FGSM, ε = 2/255 | 21.39% |
| FGSM, ε = 4/255 | 8.68% |
| FGSM, ε = 8/255 | 3.85% |
| PGD-10, ε = 8/255 | 0.00% |
| PGD-50, ε = 8/255, five restarts | 0.00% |

For the FGSM evaluations, the independently audited ASR values were 75.71%, 90.14%, and 95.70% at ε values of 2/255, 4/255, and 8/255, respectively. The PGD-10 ASR was 100%. For PGD-50 with five restarts, cumulative ASR was also 100%.

The independent ASR audit confirmed the clean-correct denominators and successful-attack counts for the validation-based standard model:

| Attack | Initially correct samples | Successfully attacked | ASR |
|---|---:|---:|---:|
| FGSM, ε = 2/255 | 8,806 | 6,667 | 75.71% |
| FGSM, ε = 4/255 | 8,806 | 7,938 | 90.14% |
| FGSM, ε = 8/255 | 8,806 | 8,427 | 95.70% |
| PGD-10, ε = 8/255 | 8,806 | 8,806 | 100.00% |

These results show that the standard model was highly vulnerable under the specified gradient-based attacks, especially the iterative PGD evaluations.

### 4.2 Effect of FGSM adversarial fine-tuning

FGSM adversarial fine-tuning increased adversarial accuracy under all reported attack settings, while reducing clean test accuracy.

| Evaluation | Standard model | FGSM-trained model |
|---|---:|---:|
| Clean accuracy | 88.06% | 68.93% |
| FGSM, ε = 2/255 | 21.39% | 60.97% |
| FGSM, ε = 4/255 | 8.68% | 52.67% |
| FGSM, ε = 8/255 | 3.85% | 38.09% |
| PGD-10, ε = 8/255 | 0.00% | 33.49% |
| PGD-50, five restarts | 0.00% | 32.25% |

The clean-accuracy reduction was 19.13 percentage points. In contrast, PGD-10 accuracy increased by 33.49 percentage points and PGD-50 robust accuracy increased by 32.25 percentage points.

The FGSM-trained model's PGD-10 ASR was 51.41%, compared with 100% for the standard model. Its PGD-50 cumulative ASR was 53.21%, compared with 100% for the standard model. The results indicate a substantial robustness improvement for the tested attacks, accompanied by a considerable reduction in clean accuracy.

### 4.3 Clean accuracy after pruning

The following table reports clean test accuracy for the validation-based standard and FGSM-trained models before and after global magnitude pruning.

| Training condition | Sparsity | Clean accuracy |
|---|---:|---:|
| Standard | 0% | 88.06% |
| Standard | 30% | 88.05% |
| Standard | 50% | 87.91% |
| Standard | 70% | 86.58% |
| FGSM-trained | 0% | 68.93% |
| FGSM-trained | 30% | 68.79% |
| FGSM-trained | 50% | 67.80% |
| FGSM-trained | 70% | 54.54% |

For the standard model, clean accuracy changed very little at 30% and 50% sparsity and decreased by 1.48 percentage points at 70% sparsity. For the FGSM-trained model, 30% and 50% pruning largely preserved clean accuracy, while 70% pruning reduced it by 14.39 percentage points relative to the unpruned FGSM-trained model.

### 4.4 Adversarial accuracy after pruning

| Training condition | Sparsity | FGSM 2/255 | FGSM 4/255 | FGSM 8/255 | PGD-10 |
|---|---:|---:|---:|---:|---:|
| Standard | 0% | 21.39% | 8.68% | 3.85% | 0.00% |
| Standard | 30% | 21.25% | 8.60% | 3.88% | 0.00% |
| Standard | 50% | 20.70% | 7.97% | 3.44% | 0.00% |
| Standard | 70% | 18.70% | 5.93% | 2.51% | 0.00% |
| FGSM-trained | 0% | 60.97% | 52.67% | 38.09% | 33.49% |
| FGSM-trained | 30% | 60.90% | 52.77% | 38.04% | 33.69% |
| FGSM-trained | 50% | 59.96% | 52.03% | 37.70% | 33.61% |
| FGSM-trained | 70% | 47.98% | 41.72% | 31.29% | 28.94% |

Pruning did not improve the standard model's PGD-10 accuracy; it remained at 0% for every tested sparsity level. The standard model's adversarial accuracy generally declined as sparsity increased.

For the FGSM-trained model, 30% and 50% pruning largely preserved the measured adversarial accuracy. At 70% sparsity, accuracy declined across all reported attack settings. The small PGD-10 accuracy differences at 30% and 50% should be interpreted as near-preservation in this single-seed experiment, not as evidence that pruning reliably improves robustness.

### 4.5 PGD-10 attack success rate after pruning

| Training condition | Sparsity | PGD-10 ASR |
|---|---:|---:|
| Standard | 0% | 100.00% |
| Standard | 30% | 100.00% |
| Standard | 50% | 100.00% |
| Standard | 70% | 100.00% |
| FGSM-trained | 0% | 51.41% |
| FGSM-trained | 30% | 51.02% |
| FGSM-trained | 50% | 50.44% |
| FGSM-trained | 70% | 47.25% |

ASR is conditional on the samples correctly classified before the attack. The lower ASR at 70% sparsity for the FGSM-trained model does not by itself mean that the model became more robust: clean accuracy and adversarial accuracy both declined substantially. The metrics must be interpreted together.

### 4.6 Preliminary inference-efficiency results

| Metric | Baseline | Fine-tuned 70%-pruned |
|---|---:|---:|
| Median latency, batch 64 | 49.8516 ms | 49.7795 ms |
| Throughput | 1,283.81 images/s | 1,285.67 images/s |
| Checkpoint size | 42.7021 MiB | 42.7051 MiB |

The latency and throughput differences were negligible in this benchmark, and the pruned checkpoint was marginally larger. The result does not demonstrate a meaningful inference-speed or storage benefit from unstructured pruning in the tested dense execution environment.

### 4.7 Supporting experiments

Earlier five-epoch SimpleCNN experiments on CIFAR-10 and CIFAR-100 are retained as supporting exploratory work. For CIFAR-10, the SimpleCNN clean baseline achieved 55.53% clean accuracy and 3.46% PGD-10 accuracy in the model-comparison run; FGSM adversarial training achieved 45.88% clean accuracy and 27.68% PGD-10 accuracy. For CIFAR-100, the clean baseline achieved 24.61% clean accuracy and 2.33% PGD-10 accuracy, while the FGSM-trained model achieved 17.84% clean accuracy and 9.86% PGD-10 accuracy.

These results are not directly comparable to the principal ResNet-18 results because they use a different architecture and shorter training procedure. They are single-seed exploratory findings.

An earlier ResNet-18 track used test-set accuracy for checkpoint selection. Its results are retained in the repository as historical experimental records, but are not used as the principal final comparison because test-set model selection introduces selection bias. The separate validation-based track is the principal result set in this report.

## 5. Discussion

The principal findings are:

1. **The standard ResNet-18 was highly vulnerable to the evaluated attacks.** Despite achieving 88.06% clean test accuracy, its adversarial accuracy was 0% under PGD-10 and PGD-50 with the reported settings.
2. **FGSM adversarial fine-tuning improved robustness under the tested attacks.** PGD-10 accuracy increased to 33.49% and PGD-50 robust accuracy to 32.25%. This improvement came with a 19.13 percentage-point clean-accuracy reduction.
3. **Moderate pruning largely preserved the measured performance of the FGSM-trained model.** The 30% and 50% pruning configurations showed relatively small changes in clean and adversarial accuracy in this experiment.
4. **Aggressive 70% pruning harmed the FGSM-trained model.** Clean accuracy decreased by 14.39 percentage points, and adversarial accuracy declined across the tested attack settings.
5. **Unstructured sparsity did not produce practical dense-inference gains in the benchmark.** The measured latency, throughput, and checkpoint size were essentially unchanged.

These observations support a cautious conclusion: the effect of pruning depends on the training condition and sparsity level, and sparsity should not be treated as a proxy for robustness or deployment efficiency. The experiments do not show that unstructured pruning is universally ineffective, nor do they establish that other compression methods cannot preserve robustness.

## 6. Limitations

1. **Single training seed:** The principal comparisons use one training seed per configuration. Multi-seed variability and statistical confidence intervals were not evaluated.
2. **One dataset and architecture for the main study:** The principal comparison uses CIFAR-10 and one CIFAR-adapted ResNet-18 architecture.
3. **Attack coverage:** The conclusions apply to the specified FGSM and PGD settings. They do not establish robustness against all attacks, threat models, or distribution shifts.
4. **ASR interpretation:** ASR is conditional on clean-correct samples and must be considered alongside clean and adversarial accuracy.
5. **Pruning scope:** The study evaluates global unstructured L1 magnitude pruning at 30%, 50%, and 70%. It does not evaluate structured pruning or specialized sparse inference.
6. **Efficiency benchmark:** The benchmark is hardware-specific and uses one GPU and batch size 64. It does not include sparse kernels, compressed sparse storage, or deployment on other hardware.
7. **CIFAR-100 results:** The supporting CIFAR-100 experiments use a five-epoch SimpleCNN and one seed; they are exploratory rather than a mature benchmark.
8. **HYDRA reproduction:** The HYDRA paper and reference implementation were inspected, and limited smoke tests were run. A full reproduction of the paper was not completed.
9. **Generalization:** The results should not be generalized to other architectures, datasets, training procedures, pruning algorithms, or deployment environments without additional experiments.

## 7. Conclusion

This project established and evaluated a validation-based CIFAR-10 ResNet-18 pipeline for studying adversarial training and unstructured magnitude pruning. The principal standard model achieved 88.06% clean test accuracy but was highly vulnerable to the evaluated PGD attacks. FGSM adversarial fine-tuning improved PGD-10 accuracy to 33.49% and PGD-50 robust accuracy to 32.25%, at the cost of reducing clean accuracy to 68.93%.

Pruning at 30% and 50% sparsity largely preserved the measured performance of the FGSM-trained model, while 70% sparsity caused a substantial decline in both clean and adversarial accuracy. The standard model remained vulnerable to PGD-10 at every tested pruning level. The preliminary GTX 1050 benchmark did not demonstrate meaningful speed or checkpoint-size benefits from unstructured pruning in dense execution.

The main contribution of this project is an experimentally documented comparison of standard training, FGSM adversarial fine-tuning, and multiple pruning levels using a validation-selected baseline, held-out test evaluation, and independently audited ASR calculations. The findings are limited by the single-seed design and the tested architecture, dataset, attacks, and hardware. They provide a reproducible starting point for further research into robustness-aware model compression, rather than a general claim that one training or pruning method is universally superior.

## Reproducibility and project artifacts

The repository contains the experiment scripts and CSV records supporting the results. Principal records include:

- `experiments/resnet18_validated_attack_results.csv`
- `experiments/resnet18_validated_asr_audit.csv`
- `experiments/resnet18_validated_fgsm_adv_comparison.csv`
- `experiments/resnet18_validated_pgd50_multirestart.csv`
- `experiments/resnet18_validated_baseline_pruning_evaluation.csv`
- `experiments/resnet18_validated_fgsm_adv_pruning_evaluation.csv`
- `experiments/resnet18_validated_combined_pruning_evaluation.csv`
- `experiments/resnet18_validated_final_pruning_summary.csv`

The repository also contains training, evaluation, pruning, plotting, and benchmarking scripts, together with the project documentation. Model checkpoints and datasets are maintained locally rather than committed as large repository files.

## References

[1] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and Harnessing Adversarial Examples,” *International Conference on Learning Representations (ICLR)*, 2015. https://arxiv.org/abs/1412.6572

[2] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards Deep Learning Models Resistant to Adversarial Attacks,” *International Conference on Learning Representations (ICLR)*, 2018. https://arxiv.org/abs/1706.06083

[3] S. Ye, K. Xu, S. Liu, H. Cheng, J. B. Lambrechts, H. Zhang, A. Zhou, K. Ma, Y. Wang, and X. Lin, “Adversarial Robustness vs. Model Compression, or Both?” arXiv:1903.12561, 2019/2021. https://arxiv.org/abs/1903.12561

[4] V. Sehwag, S. Wang, P. Mittal, and S. Jana, “HYDRA: Pruning Adversarially Robust Neural Networks,” *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. https://arxiv.org/abs/2002.10509
