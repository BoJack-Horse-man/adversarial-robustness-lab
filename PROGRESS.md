
# Research Progress Log

**Project:** Adversarial Robustness of Efficient Neural Networks  
**Status:** Initial experiments completed; research ongoing  
**Last updated:** 2026-10-20

## 1. Current Position

### Completed
- [x] Define the research goal and broad milestones.
- [x] Select HYDRA as the first reference repository.
- [x] Confirm the local computer and development setup.
- [x] Create and connect the personal GitHub repository.
- [x] Set up the Python environment.
- [x] Download and verify CIFAR-10.
- [x] Train a clean baseline model.
- [x] Implement and evaluate FGSM and PGD attacks.
- [x] Run FGSM adversarial-training experiments.
- [x] Compare baseline and adversarially trained models.

### Remaining
- [ ] Prepare and evaluate CIFAR-100.
- [ ] Study the HYDRA paper and implementation.
- [ ] Reproduce a suitable HYDRA experiment.
- [ ] Investigate pruning and robustness trade-offs.
- [ ] Improve experimental reproducibility with multiple random seeds.
- [ ] Measure model size, runtime, and computational efficiency.

## 2. Session Log

### 2026-10-19: Research Planning

**Completed**
- Defined a staged workflow from environment setup to pruning research.
- Chose CIFAR-10 as the first working dataset and CIFAR-100 as the follow-up.
- Selected HYDRA as the first codebase to study.

**Decisions**
- Work one step at a time and verify each step before proceeding.
- Keep our code in a separate repository from reference implementations.
- Start with a baseline model before studying HYDRA's older dependencies in isolation.
- Record experiment settings and results instead of relying on memory.

### 2026-10-20: Environment Setup

**Environment**
- Operating system: Windows 10.
- Python: 3.10.22.
- Conda environment: `adv-robust`.
- PyTorch: 2.5.1+cu121.
- TorchVision: 0.20.1+cu121.
- NumPy: 2.2.6.
- Matplotlib: 3.10.9.
- tqdm: 4.70.1.
- GPU: NVIDIA GeForce GTX 1050, 4 GB VRAM.

**Verification**
- [x] Verified that `torch.cuda.is_available()` returns `True`.
- [x] Successfully performed matrix multiplication on `cuda:0`.
- [x] Initialized the GitHub repository and connected the remote repository.

### First CIFAR-10 Baseline

**Dataset**
- Dataset: CIFAR-10.
- Training images: 50,000.
- Test images: 10,000.
- Transformation: `ToTensor()`, scaling pixel values to [0, 1].

**Model and training**
- Architecture: SimpleCNN.
- Trainable parameters: 94,538.
- Optimizer: Adam.
- Learning rate: 0.001.
- Batch size: 64.
- Epochs: 5.
- Random seed: 42.

**Results**
- Epoch 1 training accuracy: 31.50%.
- Epoch 5 training accuracy: 53.85%.
- Epoch 5 average training loss: 1.2739.
- Clean test accuracy: 55.53%.

**Artifacts**
- Training script: `train_baseline.py`.
- Dataset check: `check_cifar10.py`.
- Dataset visualization: `inspect_cifar10.py`.
- Results: `experiments/baseline_result.txt`.
- Model weights: `checkpoints/simple_cnn.pth` (local; not tracked by Git).

**Note:** This is a preliminary five-epoch baseline, not a final benchmark or a multi-seed result.

### FGSM Attack Evaluation

**Setup**
- Model: SimpleCNN trained for five epochs.
- Dataset: CIFAR-10 test set (10,000 images).
- Attack: Untargeted Fast Gradient Sign Method (FGSM).
- Pixel range: [0, 1].
- Checkpoint: `checkpoints/simple_cnn.pth`.

**Results**

| Condition | Epsilon | Test accuracy |
|---|---:|---:|
| Clean | 0 | 55.53% |
| FGSM | 2/255 | 34.57% |
| FGSM | 4/255 | 20.54% |
| FGSM | 8/255 | 7.61% |

**Interpretation**

Test accuracy decreased as the FGSM perturbation strength increased. At epsilon = 8/255, accuracy fell to 7.61%, indicating that this baseline model is vulnerable to FGSM under the tested settings.

These are preliminary results from one trained model and one random seed. They do not establish general robustness conclusions.

**Artifact**
- Script: `eval_fgsm.py`.
- Results: `experiments/fgsm_results.csv`.

### PGD Attack Evaluation

**Experimental setup**
- Dataset: CIFAR-10 test set (10,000 images).
- Model: SimpleCNN trained for five epochs.
- Device: NVIDIA GPU (`cuda`).
- Attack: Untargeted Projected Gradient Descent (PGD).
- Maximum perturbation: ε = 8/255.
- Step size: α = 2/255.
- Iterations: 10.
- Random start: Enabled.
- Pixel range: [0, 1].

**Standalone evaluation results**

| Evaluation | Test accuracy |
|---|---:|
| Clean images | 55.53% |
| FGSM, ε = 8/255 | 7.61% |
| PGD, ε = 8/255, 10 steps | 3.49% |

PGD reduced accuracy by 52.04 percentage points compared with clean evaluation. Under these settings, PGD reduced accuracy further than FGSM at the same maximum perturbation.

**Interpretation**

The model is vulnerable to gradient-based adversarial attacks. However, its clean accuracy is only 55.53%, so further experiments are needed before drawing broader conclusions about robustness.

**Artifact**
- Script: `eval_pgd.py`.
- Results: `experiments/pgd_results.csv`.

### FGSM Adversarial Training and Model Comparison

**Setup**
- Dataset: CIFAR-10.
- Training method: FGSM adversarial training.
- Training perturbation: ε = 8/255.
- Epochs: 5.
- Batch size: 64.
- Optimizer: Adam.
- Learning rate: 0.001.
- The adversarially trained model was trained separately from scratch.

**Baseline versus adversarially trained model**

| Evaluation | Baseline | Adversarially trained |
|---|---:|---:|
| Clean images | 55.53% | 45.88% |
| FGSM, 2/255 | 34.57% | 41.39% |
| FGSM, 4/255 | 20.54% | 37.12% |
| FGSM, 8/255 | 7.61% | 28.87% |
| PGD, 8/255, 10 steps | 3.46% | 27.68% |

**Main observations**
- Adversarial training improved accuracy under all tested FGSM perturbation strengths.
- It also improved accuracy under the tested PGD configuration.
- Clean test accuracy decreased from 55.53% to 45.88%, a reduction of 9.65 percentage points.
- The baseline PGD result was 3.49% in the standalone run and 3.46% in the model-comparison run. These values came from separate runs with random PGD initialization.

**Interpretation**

The preliminary results suggest a trade-off between clean accuracy and adversarial robustness. Adversarial training improved robust accuracy in the tested experiments but reduced clean accuracy. More experiments are needed to assess reproducibility and generalization.

**Artifacts**
- Training script: `train_adversarial.py`.
- Evaluation script: `eval_model_comparison.py`.
- Training summary: `experiments/adversarial_training_result.txt`.
- Training history: `experiments/adversarial_training_history.csv`.
- Comparison results: `experiments/model_robustness_comparison.csv`.

## 3. Consolidated Experiment Results

Dataset: CIFAR-10 test set (10,000 images).

The detailed experimental setups and interpretations are documented in the session log above. The key accuracy results are summarized here.

| Model | Clean | FGSM 2/255 | FGSM 4/255 | FGSM 8/255 | PGD 8/255 |
|---|---:|---:|---:|---:|---:|
| Baseline SimpleCNN | 55.53% | 34.57% | 20.54% | 7.61% | 3.46%* |
| FGSM adversarial training | 45.88% | 41.39% | 37.12% | 28.87% | 27.68% |

*The baseline PGD accuracy in the comparison experiment was 3.46%; the separate PGD evaluation returned 3.49%.

**Result files**
- `experiments/baseline_result.txt`
- `experiments/fgsm_results.csv`
- `experiments/pgd_results.csv`
- `experiments/adversarial_training_result.txt`
- `experiments/adversarial_training_history.csv`
- `experiments/model_robustness_comparison.csv`

## 4. Current Limitations

- The baseline is a small custom CNN, not ResNet-18 or another standard benchmark architecture.
- Training was limited to five epochs.
- Important experiments have not yet been repeated across multiple random seeds.
- PGD evaluation should be further checked for reproducibility and attack strength.
- CIFAR-100 has not yet been evaluated.
- Pruning and quantization have not yet been implemented.
- No HYDRA experiment has been reproduced.
- Model size, inference latency, and computational savings have not yet been measured.

## 5. Next Steps

### Priority 1: Validate Existing Experiments
1. Review the FGSM and PGD implementations.
2. Improve evaluation reproducibility where possible.
3. Repeat key experiments with multiple random seeds.
4. Record attack settings consistently.
5. Investigate stronger evaluation settings and potential robustness evaluation pitfalls.

### Priority 2: Study HYDRA
1. Read *HYDRA: Pruning Adversarially Robust Neural Networks*.
2. Understand its motivation and main contribution.
3. Study its pretraining, pruning, and fine-tuning stages.
4. Compare its pruning approach with ordinary magnitude-based pruning.
5. Inspect the implementation before attempting a reproduction.

**References**
- Paper: https://arxiv.org/abs/2002.10509
- Official repository: https://github.com/inspire-group/hydra

### Priority 3: Extend the Experiments
1. Prepare and evaluate CIFAR-100.
2. Consider a standard architecture such as ResNet-18, subject to GPU memory limits.
3. Implement and evaluate a suitable pruning method.
4. Measure parameter count, model size, accuracy, robustness, and inference time.
5. Compare clean and robust accuracy before and after pruning.
6. Explore quantization if resources permit.

### Priority 4: Identify a Research Gap
1. Review related work on adversarial robustness and model compression.
2. Compare the strengths and limitations of existing methods.
3. Identify a research question that can be tested with the available hardware.
4. Discuss the proposed research question and experimental design with the supervisor.

## 6. Research Decisions and Workflow

- Work one step at a time and verify each step before proceeding.
- Keep our implementation separate from reference repositories.
- Record experimental settings and results in the project.
- Treat current findings as preliminary until they have been tested for reproducibility.
- Base the eventual research contribution on a clearly defined research gap and reproducible experiments.
