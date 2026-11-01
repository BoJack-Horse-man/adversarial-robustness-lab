
# Adversarial Robustness Lab

A step-by-step learning and research project on adversarial robustness and efficient neural networks.

The project investigates how neural networks respond to adversarial attacks and whether adversarial training and model compression can improve robustness while maintaining good classification accuracy.

## Research Roadmap

- [x] Environment and GitHub setup
- [x] CIFAR-10 data pipeline and dataset inspection
- [x] Train a clean SimpleCNN baseline
- [x] Evaluate FGSM and PGD attacks
- [x] Implement FGSM adversarial training
- [x] Compare baseline and adversarially trained models
- [x] Prepare and evaluate CIFAR-100
- [x] Study the HYDRA paper and implementation
- [x] Reproduce a suitable HYDRA experiment
- [ ] Investigate pruning and robustness trade-offs
- [ ] Identify a research question for further experiments

## Current Experimental Results

Dataset: CIFAR-10 test set (10,000 images).

| Evaluation | Baseline SimpleCNN | FGSM adversarial training |
|---|---:|---:|
| Clean accuracy | 55.53% | 45.88% |
| FGSM, ε = 2/255 | 34.57% | 41.39% |
| FGSM, ε = 4/255 | 20.54% | 37.12% |
| FGSM, ε = 8/255 | 7.61% | 28.87% |
| PGD, ε = 8/255, 10 steps | 3.46%* | 27.68% |

The initial experiments suggest that adversarial training improves accuracy under the tested attacks, while reducing clean test accuracy. These are preliminary results from five-epoch training runs and require further validation.

*The baseline PGD result was 3.46% in the model-comparison experiment and 3.49% in a separate evaluation.*

## Project Documentation

- [PROGRESS.md](PROGRESS.md) — session history, experiment details, results, limitations, and next steps.
- [PLAN.md](PLAN.md) — research milestones and exit criteria.

## Research References

- [HYDRA: Pruning Adversarially Robust Neural Networks](https://github.com/inspire-group/hydra)
- [Robustness-Aware-Pruning-ADMM](https://github.com/yeshaokai/Robustness-Aware-Pruning-ADMM)
- [CAGE](https://github.com/XinweiZhang1998/CAGE)

## Environment

- Python 3.10.22
- PyTorch 2.5.1+cu121
- TorchVision 0.20.1+cu121
- NVIDIA GeForce GTX 1050, 4 GB VRAM
- Conda environment: `adv-robust`

## Repository Notes

Datasets, model checkpoints, credentials, and large generated files should not be committed to Git. Experiment scripts and result records are maintained in the repository where appropriate.
