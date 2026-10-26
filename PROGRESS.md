
# Research Progress Log

**Project:** Adversarial Robustness of Efficient Neural Networks  
**Status:** Initial experiments completed; research ongoing  
**Last updated:** 2026-10-25

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
- [x] Prepare and validate CIFAR-100 (50,000 training and 10,000 test images).
- [x] Train a CIFAR-100 clean baseline.
- [x] Implement and evaluate FGSM and PGD attacks on CIFAR-100.
- [x] Train a CIFAR-100 model using FGSM adversarial training.
- [x] Evaluate FGSM and PGD robustness after adversarial training.


### PGD-50 Multi-Restart Model Comparison - 2026-10-26

**Status:** Completed; results are preliminary.

**Dataset and evaluation**
- Dataset: CIFAR-10 test set (10,000 images).
- Models: baseline `SimpleCNN` and FGSM adversarially trained `SimpleCNN`.
- Checkpoints: `checkpoints/simple_cnn.pth` and `checkpoints/simple_cnn_adv.pth`.
- Attack: PGD-50 with five random restarts.
- Perturbation budget: epsilon `8/255` (approximately `0.03137255`).
- Step size: alpha `2/255` (approximately `0.00784314`).
- Batch size: 64.
- Device: CUDA.
- Pixel range: `[0, 1]`.
- Results file: `experiments/pgd50_5restarts_model_comparison.csv`.
- Script: `compare_pgd50_models.py`.

**Results**

| Model | Clean accuracy | PGD-50 robust accuracy |
|---|---:|---:|
| Baseline CNN | 55.53% | 2.42% |
| FGSM adversarially trained CNN | 45.88% | 27.35% |

The adversarially trained model improved PGD-50 robust accuracy by 24.93 percentage points, while clean accuracy decreased by 9.65 percentage points compared with the baseline.

**Interpretation and limitations**
- Under this evaluation, FGSM adversarial training substantially improved the model's resistance to PGD-50 attacks.
- The clean-accuracy reduction shows a trade-off between clean performance and adversarial robustness.
- Results come from one training run per model. Multiple training seeds and further attack validation are needed before drawing strong conclusions.
- The experiment evaluates adversarial robustness, but does not yet establish whether pruning or quantization can preserve robustness efficiently.

**Next steps**
1. Validate the evaluation protocol and compare results across multiple training seeds.
2. Establish a controlled pruning baseline and measure clean accuracy, robust accuracy, sparsity, model size, and inference latency.
3. Evaluate quantization separately.
4. Compare compressed models with and without adversarial training under matched evaluation settings.

### Remaining
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

### Multi-Restart PGD-50 Evaluation

**Experimental setup**
- Dataset: CIFAR-10 test set (10,000 images).
- Model: SimpleCNN clean baseline, trained for five epochs with random seed 42.
- Device: NVIDIA GPU (`cuda`).
- Attack: Untargeted Projected Gradient Descent (PGD).
- Maximum perturbation: epsilon = 8/255.
- Step size: alpha = 2/255.
- Iterations per restart: 50.
- Random restarts: 5.
- Pixel range: [0, 1].

**Results**

| Evaluation | Test accuracy |
|---|---:|
| Clean images | 55.53% |
| PGD-50, 5 random restarts | 2.42% |

The multi-restart evaluation took approximately 4 minutes and 34 seconds on the available GPU.

**Interpretation**

Robust accuracy decreased from 2.54% in the single-start PGD-50 run with seed 42 to 2.42% with five random restarts. This is consistent with multiple restarts finding additional successful adversarial examples. The result indicates substantial vulnerability under the tested attack configuration, although the model's clean accuracy is also limited.

These results are preliminary and use one trained model. They do not establish general conclusions about other architectures, training seeds, datasets, or compression methods.

**Artifacts**
- Script: `eval_pgd_restarts.py`.
- Results: `experiments/pgd50_5restarts_results.csv`.
- Related plots: `experiments/PGD adversarial accuracy by attack steps.png` and `experiments/PGD-50 accuracy by seed.png`.

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

The detailed experimental setups and interpretations are documented in the session log above. The key accuracy results for CIFAR-10 and CIFAR-100 are summarized below.

### CIFAR-10

Dataset: CIFAR-10 test set (10,000 images).

| Model | Clean | FGSM 2/255 | FGSM 4/255 | FGSM 8/255 | PGD 8/255 |
|---|---:|---:|---:|---:|---:|
| Baseline SimpleCNN | 55.53% | 34.57% | 20.54% | 7.61% | 3.46%* |
| FGSM adversarial training | 45.88% | 41.39% | 37.12% | 28.87% | 27.68% |

*The baseline PGD accuracy in the comparison experiment was 3.46%; the separate PGD evaluation returned 3.49%.

### CIFAR-100

Dataset: CIFAR-100 test set (10,000 images; 100 classes).

| Model | Clean | FGSM 2/255 | FGSM 4/255 | FGSM 8/255 | PGD 8/255 |
|---|---:|---:|---:|---:|---:|
| Baseline SimpleCNN | 24.61% | 15.60% | 9.43% | 3.80% | 2.33% |
| FGSM adversarial training | 17.84% | 15.70% | 13.73% | 10.33% | 9.86% |

### Preliminary observations

- On CIFAR-100, FGSM adversarial training increased PGD-10 accuracy from 2.33% to 9.86%, an improvement of 7.53 percentage points.
- Clean accuracy decreased from 24.61% to 17.84%, a reduction of 6.77 percentage points.
- At FGSM ε=8/255, accuracy increased from 3.80% to 10.33%.
- These results are preliminary: both models were trained for five epochs using one random seed. Multiple-seed experiments, longer training, and further robustness evaluation are needed before drawing firm conclusions.

### Result files

**CIFAR-10**
- `experiments/baseline_result.txt`
- `experiments/fgsm_results.csv`
- `experiments/pgd_results.csv`
- `experiments/adversarial_training_result.txt`
- `experiments/adversarial_training_history.csv`
- `experiments/model_robustness_comparison.csv`

**CIFAR-100**
- `experiments/cifar100_baseline_result.txt`
- `experiments/cifar100_fgsm_results.csv`
- `experiments/cifar100_pgd_results.csv`
- `experiments/cifar100_adversarial_training_result.txt`
- `experiments/cifar100_adversarial_training_history.csv`
- `experiments/cifar100_fgsm_adv_training_results.csv`
- `experiments/cifar100_pgd_adv_training_results.csv`

## 4. Current Limitations

- The baseline is a small custom CNN, not ResNet-18 or another standard benchmark architecture.
- Training was limited to five epochs on both CIFAR-10 and CIFAR-100.
- Important experiments have not yet been repeated across multiple random seeds.
- The CIFAR-100 results are preliminary and need validation with longer training runs and multiple seeds.
- PGD evaluation should be further checked for reproducibility, attack strength, and possible evaluation pitfalls.
- Pruning and quantization have not yet been implemented.
- HYDRA diagnostics have been performed, but a full research experiment has not been reproduced.
- Model size, inference latency, and computational savings have not yet been measured.
- The current results do not yet establish the trade-off between model compression and adversarial robustness.

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
1. Validate CIFAR-100 results across multiple random seeds and longer training runs.
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

## HYDRA Reference-Code Diagnostics - 2026-10-24

**Status:** Smoke tests completed; full paper reproduction not completed.

These experiments use the separate reference repository at
`E:\SUST\super\Experiment\hydra-reference`. Data and results are stored separately from the main project:

- Data: `E:\SUST\super\Experiment\hydra-pilot-data`
- Results: `E:\SUST\super\Experiment\hydra-pilot-results`
- Reference paper: *HYDRA: Pruning Adversarially Robust Neural Networks* (NeurIPS 2020), https://arxiv.org/abs/2002.10509

### Environment and scope

- Dataset: CIFAR-10, using a 10% training-data fraction for smoke tests.
- Architecture: lightweight `cifar_model`.
- Hardware: NVIDIA GeForce GTX 1050, 4 GB VRAM.
- These short runs test whether the reference code and experiment pipeline execute. They are not a faithful reproduction of the paper.

### Diagnostic results

| Experiment | Best/final reported validation accuracy | Weight sparsity |
|---|---:|---:|
| Dense baseline, Run 7 | 24.87% | 0% |
| Subnet pretraining, before pruning | 17.97% | Nominally 0% at k=1.0 |
| Pruning run, dense export | 27.83% | 50% |

Settings varied by run. The dense baseline and subnet runs used two epochs; the pruning diagnostic used two epochs with `k=0.5` and `--save-dense`. The pruning checkpoint was evaluated using `--layer-type dense`.

**Interpretation caution:** These numbers are not a fair controlled comparison. Training histories and procedures differ, only 10% of CIFAR-10 training data was used, and the pruning stage continued training the pruning scores. The 27.83% result must not be interpreted as evidence that pruning improves accuracy.

### Checkpoint and sparsity verification

- Dense baseline checkpoint evaluated at 24.87%, matching its saved best-accuracy metadata.
- Subnet checkpoint contained `popup_scores` parameters.
- Pruning-stage diagnostic reported 50% pruning.
- Exported `model_best_dense.pth.tar` contained 214,760 weight parameters, of which 107,380 were zero: verified weight sparsity of 50%.
- The pruning-stage sanity check reported weight updates disabled and score updates enabled, as expected for that stage.

### PGD implementation smoke test

A small synthetic-batch test called HYDRA's `pgd_whitebox()` implementation with epsilon `8/255`, three steps, step size `2/255`, random start enabled, and pixel range `[0,1]`.

Checks passed:
- Output shape remained `(4, 3, 32, 32)`.
- Adversarial pixels remained within `[0,1]`.
- Maximum perturbation was approximately `0.0313726`, within epsilon `8/255`.
- All output values were finite.

This was a numerical smoke test on random inputs and labels. It did **not** measure attack success rate or adversarial accuracy.

### Reference-code caveats

- `train.py` has a nonfatal logging-formatting issue around the dataset-size log message.
- `utils/eval.py` was patched from `.view(-1)` to `.reshape(-1)` to avoid a tensor-layout error.
- `utils/model.py` was patched to return `(False, False)` when `sanity_check_paramter_updates` encounters a dense model without `popup_scores`. Treat this fallback cautiously because it may not accurately diagnose parameter updates for every dense model.
- A pruning run without `--save-dense` failed when the code attempted to report the pruning fraction without a dense checkpoint. The diagnostic run was repeated with `--save-dense`.

These changes and caveats belong to the separate reference repository, not the main project.

### Next steps

1. Review the reference-code patches and document them reproducibly.
2. Run controlled experiments with matched data, architecture, seeds, training budgets, and evaluation procedures.
3. Evaluate clean accuracy and adversarial accuracy using the same attack settings for every model.
4. Verify sparsity and checkpoint loading for each run.
5. Only then assess whether a meaningful comparison or research question can be supported.

### HYDRA subnet PGD evaluation - 2026-10-24

**Status:** Evaluation completed; results are preliminary.

Checkpoint:
`hydra-pilot-results/cifar10-subnet-pretrain/pretrain/latest_exp/checkpoint/model_best.pth.tar`

Configuration:
- Dataset: CIFAR-10 test set (10,000 images).
- Architecture: `cifar_model`, subnet, `k=1.0`.
- Checkpoint: best subnet pretraining checkpoint, loaded at epoch 1.
- Attack: HYDRA `pgd_whitebox()`, 10 steps, random start enabled.
- Perturbation budget: epsilon `8/255` (approximately `0.03137255`).
- Step size: `2/255` (approximately `0.00784314`).
- Pixel range: `[0, 1]`.

Results:

| Metric | Clean inputs | PGD adversarial inputs |
|---|---:|---:|
| Top-1 accuracy | 17.97% | 9.49% |
| Top-5 accuracy | 70.83% | 49.44% |

Top-1 accuracy decreased by 8.48 percentage points under this evaluation, approximately a 47.2% relative decrease from clean accuracy.

**Interpretation and limitations:**
- The checkpoint was trained for only two epochs using 10% of the CIFAR-10 training data; these are pilot results, not a full HYDRA reproduction.
- The existing HYDRA PGD implementation has methodological limitations that should be reviewed, including its random-start handling before the first gradient step.
- The accuracy decrease alone does not establish a reliable robustness comparison. The attack implementation and evaluation protocol should be independently checked before drawing conclusions.
- A logging-formatting error appeared during startup, but the checkpoint loaded and all 157 test batches completed successfully.

### Next evaluation actions

1. Review and validate the PGD implementation, including random-start clipping and perturbation constraints.
2. Re-evaluate clean and adversarial accuracy with a verified attack implementation and identical test examples.
3. Save the attack configuration and results in a structured CSV file.
4. Compare models only under matched training budgets, data fractions, and evaluation settings.

### Global Magnitude Pruning Baseline - 2026-10-26

**Status:** Initial pruning and robustness comparison completed; results are preliminary.

**Objective:** Measure the effect of 50% global unstructured magnitude pruning on clean accuracy and adversarial robustness of the existing SimpleCNN baseline.

#### Configuration

- Dataset: CIFAR-10 test set (10,000 images).
- Architecture: SimpleCNN, 94,538 total parameters.
- Original checkpoint: `checkpoints/simple_cnn.pth`.
- Pruned checkpoint: `checkpoints/simple_cnn_pruned_50.pth`.
- Pruning method: global unstructured L1 (magnitude) pruning.
- Target weight sparsity: 50%.
- Eligible weights: convolutional and linear layer weights; biases were not pruned.
- Pruning was applied to the existing checkpoint without retraining.
- Preprocessing: `transforms.ToTensor()`, with pixel values in `[0,1]`.
- Clean evaluation batch size: 64.
- Adversarial evaluation: PGD-50, five random restarts.
- Perturbation budget: epsilon `8/255`.
- Step size: `2/255`.
- Device: CUDA.

#### Verified sparsity and checkpoint size

| Metric | Result |
|---|---:|
| Total model parameters | 94,538 |
| Eligible weights | 94,304 |
| Zero eligible weights after pruning | 47,152 |
| Measured weight sparsity | 50.00% |
| Original checkpoint size | 381,668 bytes |
| Pruned checkpoint size | 381,788 bytes |

The pruned checkpoint was 120 bytes larger than the original. The state dictionary stores dense tensors, so setting weights to zero did not reduce storage size.

#### Results

| Metric | Baseline | 50% pruned |
|---|---:|---:|
| Clean accuracy | 55.53% | 53.82% |
| PGD-50 robust accuracy, five restarts | 2.42% | 2.32% |

- Clean accuracy change: -1.71 percentage points.
- PGD-50 robust accuracy change: -0.10 percentage points.

Result files:
- `experiments/pruning_baseline_results.csv`
- `experiments/pruning_clean_comparison.csv`
- `experiments/pruned_pgd50_5restarts_comparison.csv`

#### Interpretation and limitations

- The pruned model retained most of its baseline clean accuracy, with a decrease of 1.71 percentage points.
- Robust accuracy was nearly unchanged in this single comparison, decreasing by 0.10 percentage points.
- These results do not show a robustness improvement from pruning.
- Only one baseline checkpoint and one pruning level have been evaluated; repeated training seeds and additional sparsity levels are needed.
- Actual inference latency and peak memory usage have not been measured.
- Unstructured weight sparsity does not automatically reduce dense checkpoint storage or accelerate ordinary dense inference.
- The evaluation uses a fixed PGD configuration. Further attack validation and evaluation across multiple seeds would strengthen the conclusions.

#### Next steps

1. Measure baseline and pruned inference latency using the same device, batches, warm-up procedure, and timing method.
2. Evaluate additional pruning levels, such as 20%, 50%, and 70%, with clearly recorded configurations.
3. Investigate pruning combined with adversarial training.
4. Repeat key experiments across multiple training seeds before making general claims.
5. Keep clean accuracy, robust accuracy, checkpoint size, and inference latency as separate evaluation metrics.
