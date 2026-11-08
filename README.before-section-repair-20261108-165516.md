# Adversarial Robustness Lab

A learning and research project on **adversarial robustness of efficient neural networks**. The experiments investigate how adversarial training and model compression affect clean classification accuracy and robustness against gradient-based attacks.

The main experimental work uses CIFAR-10 and CIFAR-100 with a small SimpleCNN, followed by a CIFAR-adapted ResNet-18 track. The current results are preliminary and are intended to support further research, not to establish general claims about robustness or efficiency.

## Research Roadmap

- [x] Set up the Python/PyTorch environment, GPU, and GitHub repository.
- [x] Prepare CIFAR-10 and CIFAR-100 data pipelines.
- [x] Train and evaluate initial SimpleCNN baselines.
- [x] Implement FGSM and PGD attacks and compare the ResNet-18 implementations with `torchattacks` for the tested settings.
- [x] Run FGSM adversarial-training experiments and compare clean/adversarial accuracy.
- [x] Evaluate 30%, 50%, and 70% global magnitude pruning on adversarially trained SimpleCNN and ResNet-18 models.
- [x] Fine-tune pruned models and compare clean accuracy, adversarial accuracy, and attack success rate (ASR).
- [x] Evaluate selected ResNet-18 models using PGD-50 with five random restarts.
- [x] Conduct a preliminary ResNet-18 inference-efficiency benchmark.
- [x] Inspect the HYDRA paper and run limited implementation smoke tests.
- [x] Retrain the ResNet-18 baseline with validation-only checkpoint selection.
- [x] Independently audit ASR calculations and evaluation denominators for the validation-based baseline.
- [x] Complete validation-based FGSM adversarial fine-tuning and matched FGSM/PGD evaluations.
- [x] Complete PGD-50 evaluation with five random restarts for the validation-based standard and FGSM-trained models.
- [x] Complete 30%, 50%, and 70% pruning evaluations for both validation-based model types.
- [x] Consolidate the validation-based pruning evaluation results.
- [ ] Finalize the consolidated results, analysis, limitations, and research report.

The experimental scope is frozen. Additional multi-seed experiments and other new experiments are not planned for the current project.

## Main Experimental Results

### Initial SimpleCNN experiments

The following are five-epoch CIFAR-10 experiments on the 10,000-image test set, using one training seed.

| Evaluation | Clean baseline | FGSM adversarially trained |
|---|---:|---:|
| Clean accuracy | 55.53% | 45.88% |
| FGSM, Îµ = 2/255 | 34.57% | 41.39% |
| FGSM, Îµ = 4/255 | 20.54% | 37.12% |
| FGSM, Îµ = 8/255 | 7.61% | 28.87% |
| PGD-10, Îµ = 8/255 | 3.46%* | 27.68% |

These initial results show a clean-accuracy and adversarial-accuracy trade-off in this short training setup. The baseline PGD result was 3.46% in the model-comparison run and 3.49% in a separate evaluation.

### CIFAR-adapted ResNet-18: Earlier Experimental Track

The following results belong to the earlier ResNet-18 experimental track. The baseline was trained on the full 50,000-image CIFAR-10 training set for 100 epochs. The CIFAR adaptation uses a 3Ã—3 first convolution with stride 1 and removes the ImageNet max-pooling layer. The baseline checkpoint was selected using test accuracy, so these results are retained as historical experimental results and are not the final validation-based comparison.

| Model | Clean accuracy | PGD-50 robust accuracy | PGD-50 cumulative ASR |
|---|---:|---:|---:|
| ResNet-18 baseline | 89.26% | 0.00% | 100.00% |
| FGSM adversarially trained control | 73.50% | 34.97% | 52.42% |
| Fine-tuned, 30% pruned | 75.45% | 31.62% | 58.09% |
| Fine-tuned, 50% pruned | 75.48% | 31.81% | 57.86% |
| Fine-tuned, 70% pruned | 75.22% | 33.45% | 55.53% |

PGD-50 configuration: Îµ = 8/255, step size = 2/255, 50 steps, and five random restarts. Cumulative ASR counts an initially correctly classified image as successfully attacked if at least one restart causes misclassification. Robust accuracy is calculated over the full test set.

The unpruned adversarially trained control had the highest PGD-50 robust accuracy among these models. The fine-tuned 70%-pruned model performed best among the fine-tuned pruned models, but did not exceed the control. These are single-seed results under one attack configuration.


### Validation-based ResNet-18 experiments

A separate ResNet-18 experimental track was completed using a training/validation split to avoid selecting the checkpoint with test-set accuracy. The CIFAR-10 training data was divided into 45,000 training images and 5,000 validation images; the 10,000-image test set was reserved for final evaluation. The best checkpoint was selected using validation accuracy.

| Evaluation | Validation-based standard model | FGSM adversarially trained model |
|---|---:|---:|
| Clean test accuracy | 88.06% | 68.93% |
| FGSM, Îµ = 2/255 | 21.39% | 60.97% |
| FGSM, Îµ = 4/255 | 8.68% | 52.67% |
| FGSM, Îµ = 8/255 | 3.85% | 38.09% |
| PGD-10, Îµ = 8/255 | 0.00% | 33.49% |
| PGD-50 robust accuracy, five restarts | 0.00% | 32.25% |
| PGD-50 cumulative ASR, five restarts | 100.00% | 53.21% |

The FGSM-trained model improved adversarial accuracy under the evaluated attacks, including a 32.25 percentage-point improvement in PGD-50 robust accuracy. This came with a 19.13 percentage-point reduction in clean test accuracy. The results show a clear clean-accuracy and adversarial-robustness trade-off in this training configuration.

The baseline ASR calculations were independently audited. ASR is calculated conditionally on samples that were correctly classified before the attack. For the PGD-50 multi-restart evaluation, cumulative ASR counts an initially correctly classified image as successfully attacked if at least one of the five restarts causes misclassification.

### Validation-based pruning results

Global L1 unstructured magnitude pruning at 30%, 50%, and 70% sparsity was evaluated separately on the validation-based standard model and the FGSM-trained model.

| Model | Sparsity | Clean accuracy |
|---|---:|---:|
| Standard control | 0% | 88.06% |
| Standard pruned | 30% | 88.05% |
| Standard pruned | 50% | 87.91% |
| Standard pruned | 70% | 86.58% |
| FGSM-trained control | 0% | 68.93% |
| FGSM-trained pruned | 30% | 68.79% |
| FGSM-trained pruned | 50% | 67.80% |
| FGSM-trained pruned | 70% | 54.54% |

For the standard model, 30% and 50% pruning caused very small clean-accuracy changes, while 70% pruning caused a larger reduction. Pruning did not improve the standard model's adversarial robustness under the evaluated attacks.

For the FGSM-trained model, 30% and 50% pruning largely preserved clean and adversarial accuracy. At 70% sparsity, clean accuracy decreased by 14.39 percentage points relative to the unpruned FGSM-trained control. Small changes in ASR should not be interpreted alone as improved robustness; clean accuracy and adversarial accuracy must also be considered.

The validation-based pruning results are available in:
- `experiments/resnet18_validated_baseline_pruning_evaluation.csv`
- `experiments/resnet18_validated_fgsm_adv_pruning_evaluation.csv`
- `experiments/resnet18_validated_combined_pruning_evaluation.csv`

### Preliminary inference-efficiency benchmark

A focused benchmark compared the baseline ResNet-18 and the fine-tuned 70%-pruned model on one NVIDIA GTX 1050, using batch size 64 and ten measurement rounds.

| Metric | Baseline | Fine-tuned 70% |
|---|---:|---:|
| Median latency | 49.8516 ms | 49.7795 ms |
| Throughput | 1283.81 images/s | 1285.67 images/s |
| Checkpoint size | 42.7021 MiB | 42.7051 MiB |

This benchmark did **not** demonstrate a meaningful inference-speed or dense checkpoint-size reduction. The pruned model stores zero-valued weights in dense tensors; no sparse inference kernel or compressed sparse storage was used. The measurement is preliminary and hardware-specific.

## Important Limitations

- The experiments use one training seed per configuration. Multi-seed validation was not included in the current project scope.
- The earlier ResNet-18 track selected its checkpoint using test accuracy; its reported 89.26% clean accuracy is subject to test-set selection bias. The separate validation-based track addresses checkpoint selection using 45,000 training images, 5,000 validation images, and a held-out 10,000-image test set.
- The earlier ResNet-18 track selected its checkpoint using test accuracy; its reported 89.26% clean accuracy is subject to test-set selection bias. The separate validation-based track addresses checkpoint selection by using 45,000 training images, 5,000 validation images, and a held-out 10,000-image test set.
- The custom FGSM/PGD implementations were compared with 	orchattacks for the tested CIFAR-10 ResNet-18 settings; this does not validate every model or attack configuration.
- ASR calculations for the validation-based standard baseline were independently audited. ASR remains conditional on initially correctly classified samples and should be interpreted alongside clean and adversarial accuracy.
- PGD-50 with five restarts is a stronger evaluation than the earlier PGD-10 runs, but it is not a formal guarantee of worst-case robustness.
- CIFAR-100 SimpleCNN results are exploratory five-epoch, single-seed results.
- HYDRA was inspected and smoke-tested on a limited subset with short runs. A full reproduction of the paper has **not** been completed.
- Unstructured weight sparsity did not translate into demonstrated storage or inference-speed savings in the preliminary dense-model benchmark.

## Project Documentation

- [PROGRESS.md](PROGRESS.md) â€” detailed session history, experiment configurations, results, limitations, and remaining work.
- [PLAN.md](PLAN.md) â€” research milestones and exit criteria.

## Research References

- [HYDRA: Pruning Adversarially Robust Neural Networks](https://arxiv.org/abs/2002.10509) Â· [Official repository](https://github.com/inspire-group/hydra)
- [Robustness-Aware-Pruning-ADMM](https://github.com/yeshaokai/Robustness-Aware-Pruning-ADMM)
- [CAGE](https://github.com/XinweiZhang1998/CAGE)

## Environment

- Windows 10
- Python 3.10.22
- PyTorch 2.5.1+cu121
- TorchVision 0.20.1+cu121
- NVIDIA GeForce GTX 1050, 4 GB VRAM
- Conda environment: `adv-robust`

## Repository Notes

Datasets, model checkpoints, credentials, and large generated files should not be committed to Git. Experiment scripts and result records are maintained in the repository where appropriate.
