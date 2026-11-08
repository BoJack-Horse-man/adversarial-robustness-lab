
# Research Progress Log

**Project:** Adversarial Robustness of Efficient Neural Networks  
**Status:** Professor's three immediate experimental priorities completed; research ongoing  
**Last updated:** 2026-11-04

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
- [x] Validate custom FGSM/PGD implementations against torchattacks for the tested CIFAR-10 ResNet-18 settings.
- [x] Run 30%, 50%, and 70% global magnitude pruning on the FGSM adversarially trained SimpleCNN and evaluate clean and PGD-10 accuracy.
- [x] Plot clean and PGD-10 accuracy against measured weight sparsity.
- [x] Fine-tune 30%, 50%, and 70% pruned adversarially trained SimpleCNN models while preserving pruning masks.
- [x] Compare clean accuracy, PGD-10 accuracy, and attack success rate before and after fine-tuning.
- [x] Generate and verify before-and-after comparison plots for clean accuracy, PGD-10 accuracy, and ASR.
- [x] Establish and train a CIFAR-adapted ResNet-18 baseline on all 50,000 CIFAR-10 training images for 100 epochs.
- [x] Record ResNet-18 training configuration, including optimizer, learning rate, batch size, epochs, and learning-rate schedule.
- [x] Evaluate ResNet-18 clean accuracy and FGSM/PGD robustness.
- [x] Compare custom FGSM/PGD implementations against `torchattacks` for the tested ResNet-18 settings.
- [x] Report clean accuracy, adversarial accuracy, and ASR for the documented attack settings.
- [x] Apply 30%, 50%, and 70% global magnitude pruning to the adversarially trained ResNet-18 and evaluate clean accuracy, adversarial accuracy, and ASR.
- [x] Fine-tune the pruned ResNet-18 models while preserving the reconstructed pruning masks.
- [x] Evaluate fine-tuned pruned ResNet-18 models with PGD-50 and five random restarts.
- [x] Conduct a preliminary inference-efficiency comparison and document its limitations.

- [x] Independently audit ASR calculations for the validation-based ResNet-18 baseline.
- [x] Study the HYDRA paper and inspect its reference implementation.
- [x] Run HYDRA pipeline smoke tests and document why they are not a formal paper reproduction.
- [x] Improve baseline methodology by retraining/selecting the ResNet-18 checkpoint without using test-set accuracy for model selection.
- [x] Complete validation-based FGSM adversarial fine-tuning.
- [x] Compare the validation-based standard and FGSM-trained models under FGSM and PGD-10 attacks.
- [x] Evaluate the validation-based standard and FGSM-trained models using PGD-50 with five random restarts.
- [x] Complete 30%, 50%, and 70% pruning evaluations for the validation-based standard baseline.
- [x] Complete 30%, 50%, and 70% pruning evaluations for the validation-based FGSM-trained model.
- [x] Consolidate the validation-based pruning evaluation results.

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

### Remaining Research Work

The main experimental work has been completed within the current research scope. No additional experiments are planned. The remaining work focuses on organizing, analyzing, documenting, and presenting the existing results.

**Documentation**
- [ ] Review and finalize `PROGRESS.md`, ensuring completed experiments and historical records are accurately documented.
- [ ] Update `PLAN.md` to reflect the completed experimental work and frozen research scope.
- [ ] Update `README.md` with the final project overview, methodology, and key findings.

**Results and Analysis**
- [ ] Organize existing experimental results into consolidated comparison tables.
- [ ] Prepare final figures using the existing CSV files and experimental results.
- [ ] Compare the standard and FGSM-adversarially-trained ResNet-18 models.
- [ ] Analyze the effects of 30%, 50%, and 70% pruning on clean accuracy and adversarial robustness.
- [ ] Explain the trade-off between clean accuracy and adversarial robustness.
- [ ] Verify that the final reported metrics agree with the existing experiment records.

**Final Research Report**
- [ ] Finalize the research methodology and experimental configurations.
- [ ] Write the results and discussion based on completed experiments.
- [ ] Explain the limitations, including the single-training-seed setting.
- [ ] Document that HYDRA was inspected and smoke-tested, not fully reproduced.
- [ ] Discuss the limitations of unstructured pruning for practical inference efficiency.
- [ ] Complete the conclusion and state the research contribution supported by the existing findings.
- [ ] Prepare the final research report for supervisor review.

**Research Scope**

The following activities are excluded from the current project scope:

- Additional multi-seed training and evaluation.
- PGD-50 evaluation of additional pruned models.
- New CIFAR-100 experiments.
- Quantization experiments.
- BatchNorm ablation experiments.
- Full HYDRA reproduction.
- Additional pruning, sparse-inference, or efficiency experiments.
- New attack or defense experiments.
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
- Maximum perturbation: Îµ = 8/255.
- Step size: Î± = 2/255.
- Iterations: 10.
- Random start: Enabled.
- Pixel range: [0, 1].

**Standalone evaluation results**

| Evaluation | Test accuracy |
|---|---:|
| Clean images | 55.53% |
| FGSM, Îµ = 8/255 | 7.61% |
| PGD, Îµ = 8/255, 10 steps | 3.49% |

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
- Training perturbation: Îµ = 8/255.
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
- At FGSM Îµ=8/255, accuracy increased from 3.80% to 10.33%.
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

The professor's three immediate experimental priorities have been addressed, but the evidence remains preliminary and has important limitations:

- Most training, pruning, and fine-tuning comparisons use one training seed and one checkpoint per configuration. Multi-seed reproducibility has not been established.
- The original CIFAR-10 ResNet-18 baseline checkpoint was selected using test accuracy during training. Its reported 89.26% clean accuracy is subject to test-set selection bias. A replacement validation-based baseline has now been trained using a reproducible 45,000/5,000 training-validation split (seed 42). The best validation accuracy was 89.70% at epoch 54, and the final held-out test accuracy was 88.06%. The test set was not used for checkpoint selection. However, this is still a single-seed result. Existing ResNet-18 adversarial-training and pruning experiments were derived from the old checkpoint and have not yet been repeated using the new validated baseline.
- CIFAR-100 SimpleCNN experiments used five epochs and one seed; they are exploratory rather than a reliable benchmark.
- FGSM/PGD implementation comparison against `torchattacks` was performed for the tested CIFAR-10 ResNet-18 settings. It does not validate every attack configuration or every model. A separate independent audit of the ASR implementation and denominators remains useful.
- PGD-50 with five restarts is stronger than the earlier PGD-10 evaluation, but it is not a formal guarantee of worst-case robustness.
- HYDRA work consists of code/data/training/evaluation smoke tests on 10% of CIFAR-10 with short runs. It is not a formal reproduction of the paper.
- The inference benchmark compared only the baseline and fine-tuned 70% model at batch size 64 on one GTX 1050, with ten measurements per model. It found no meaningful latency or dense checkpoint-size improvement. No sparse inference kernel or compressed sparse storage was tested.
- Unstructured weight sparsity should not be equated with actual storage, memory, or runtime savings.
- No quantization experiment has been conducted.

## 5. Next Steps

### Priority 1: Strengthen experimental validity
1. Repeat key training, pruning, and fine-tuning experiments with multiple independent seeds.
2. Repeat ResNet-18 adversarial training and pruning experiments using the new validation-based baseline, with matched training configurations and evaluation settings.
3. Independently audit ASR calculations, including the initially-correct denominator, and verify them against per-example predictions.
4. If robustness claims are central, evaluate matched models with consistent stronger attack settings and multiple restarts.

### Priority 2: Define the research contribution
1. Compare the completed findings with the five reviewed papers and other directly relevant literature.
2. Identify a specific, defensible research gap in adversarial robustness and efficient neural networks.
3. Formulate a testable research question and discuss the proposed contribution and scope with the supervisor.

### Priority 3: Complete the research outputs
1. Consolidate training configurations, clean/adversarial accuracy, ASR, measured sparsity, checkpoint size, and inference timing into consistent result tables and figures.
2. Explain the clean-accuracy/robustness trade-offs without claiming that unstructured pruning improves robustness or inference speed where the current results do not support that claim.
3. Prepare the methodology, results, discussion, limitations, and conclusion for the supervisor.
4. Consider a controlled HYDRA reproduction or quantization only if either is necessary to answer the final research question and can be completed with the available resources.

**HYDRA references**
- Paper: https://arxiv.org/abs/2002.10509
- Official repository: https://github.com/inspire-group/hydra

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

### HYDRA PGD Implementation Comparison - 2026-10-26

**Status:** Full CIFAR-10 test-set comparison completed; findings are diagnostic and preliminary.

**Configuration**
- Dataset: CIFAR-10 test set (10,000 images).
- Model: HYDRA subnet pretraining checkpoint.
- Device: CUDA.
- Attack: 10-step PGD.
- Perturbation budget: epsilon = 8/255.
- Step size: alpha = 2/255.
- Batch size: 16.
- Random seed: 2026.
- Pixel range: [0, 1].
- Both attack implementations used the same raw random-start images for each batch.
- Results: `hydra_pgd_full_comparison_audit.csv`.
- Script: `compare_hydra_pgd_full.py`.

**Results**

| Metric | Original HYDRA-style PGD | Corrected PGD |
|---|---:|---:|
| Clean accuracy | 17.97% | 17.97% |
| Adversarial accuracy | 9.49% | 9.49% |
| Maximum perturbation | 0.03137258 | 0.03137258 |

- Both implementations processed all 10,000 test images.
- Final adversarial pixels were within [0, 1], and all final values were finite.
- All 625 batches contained some differences between the generated adversarial images.
- Predictions differed on 8 images (0.08%).
- Correct-versus-incorrect classification outcomes differed on 0 images.

**Interpretation and limitations**

The two implementations produced the same adversarial accuracy and the same classification correctness outcomes on this test set, despite producing slightly different adversarial images. Matching accuracy does not prove that the attacks are algorithmically identical or equally strong in every setting.

The original HYDRA implementation does not clamp random-start pixels to the valid image range before its first gradient update. The corrected implementation handles the initial point differently. The maximum measured perturbation exceeded the nominal epsilon by approximately 2.8e-8 due to floating-point precision, which is negligible at this scale.

The checkpoint's clean accuracy was only 17.97%, and it was trained in a limited pilot setup. These results are an implementation diagnostic, not a full reproduction of the HYDRA paper or evidence of general robustness.

**Next steps**
1. Review the comparison script and attack implementation details.
2. Confirm the evaluation protocol and document any remaining differences.
3. Use a verified, consistent attack implementation for future controlled model comparisons.
4. Preserve the original results CSV and use the audit CSV as a separate artifact.

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

### 2026-10-26: Latency Benchmark and Final Report Validation

**Completed**
- [x] Benchmark baseline and 50%-pruned SimpleCNN inference latency on CUDA.
- [x] Validate CIFAR-100 clean, FGSM, and PGD-10 results against saved CSV files.
- [x] Validate CIFAR-10 PGD-50 five-restart pruning comparison.
- [x] Review the final research report figures and references.
- [x] Add the CIFAR-100 FGSM adversarial-training result to `FINAL_RESEARCH_REPORT.md`.
- [x] Commit and push the report update (`326bb2c`).

**Latency benchmark results**

| Batch size | Baseline latency per batch | Pruned latency per batch |
|---|---:|---:|
| 1 | 0.8415 ms | 0.8272 ms |
| 64 | 5.9306 ms | 6.2054 ms |

The pruned model was approximately 1.7% faster at batch size 1 and 4.6% slower at batch size 64. Each configuration was measured in a single benchmark run, so repeated measurements are needed before drawing firm performance conclusions.

**Additional validated results**
- CIFAR-100 PGD-10 accuracy: 2.33% baseline and 9.86% after FGSM adversarial training.
- CIFAR-100 FGSM accuracy at epsilon = 8/255: 3.80% baseline and 10.33% after FGSM adversarial training.
- CIFAR-100 clean accuracy: 24.61% baseline and 17.84% after FGSM adversarial training.
- CIFAR-10 PGD-50 accuracy with five random restarts: 2.42% baseline and 2.32% after 50% pruning.

**Current conclusion**

The experiments show a clean-accuracy versus adversarial-robustness trade-off for the tested adversarial-training configuration. The pruning experiment did not demonstrate improved robustness or a consistent inference-speed benefit. These findings remain preliminary and require repeated measurements, multiple training seeds, and further attack validation.

**Next steps**
1. Review and validate the HYDRA PGD implementation.
2. Repeat latency measurements to assess variability.
3. Investigate additional pruning levels and pruning combined with adversarial training.
4. Improve reproducibility with multiple training seeds.


### Adversarially Trained SimpleCNN: Global Magnitude Pruning - 2026-11-01

**Status:** Initial pruning and PGD-10 evaluation completed; results are preliminary.

**Objective**

Measure the effect of global unstructured magnitude pruning on clean accuracy and PGD-10 robustness of the existing FGSM adversarially trained SimpleCNN.

**Configuration**
- Dataset: CIFAR-10 test set (10,000 images).
- Architecture: SimpleCNN; 94,538 total parameters.
- Source checkpoint: `checkpoints/simple_cnn_adv.pth`.
- Training provenance: FGSM adversarial training for five epochs, epsilon = 8/255, batch size 64, Adam learning rate 0.001, random seed 42.
- Pruning method: global unstructured L1 magnitude pruning over convolutional and linear layer weights.
- Target sparsity levels: 30%, 50%, and 70%, plus the unpruned 0% control.
- Pruning was applied directly to the trained checkpoint without fine-tuning.
- Pruning masks were permanently applied using `prune.remove()`.
- Evaluation preprocessing: `transforms.ToTensor()`, with pixel values in [0, 1].
- Evaluation batch size: 64.
- Attack: untargeted PGD-10 with random start.
- Perturbation budget: epsilon = 8/255.
- Step size: alpha = 2/255.
- Device: CUDA.
- Random seed reset to 42 before each model evaluation.
- Pruning script: `prune_adversarial_model.py`.
- Evaluation script: `eval_adv_pruning.py`.
- Plotting script: `plot_adv_pruning_results.py`.

**Measured sparsity**

| Model | Target sparsity | Actual weight sparsity |
|---|---:|---:|
| Adversarially trained control | 0% | 0.0000% |
| Pruned | 30% | 29.9998% |
| Pruned | 50% | 50.0000% |
| Pruned | 70% | 70.0002% |

The eligible convolutional and linear weights totalled 94,304.

**Clean and adversarial evaluation results**

| Model | Actual sparsity | Clean accuracy | PGD-10 accuracy | ASR |
|---|---:|---:|---:|---:|
| Adversarially trained control | 0.0000% | 45.88% | 27.66% | 40.26% |
| 30% pruned | 29.9998% | 45.47% | 27.55% | 39.78% |
| 50% pruned | 50.0000% | 45.30% | 27.44% | 39.78% |
| 70% pruned | 70.0002% | 44.87% | 27.24% | 39.98% |

ASR is the percentage of initially correctly classified test images that were misclassified after the attack.

**Main observations**
- At 70% weight sparsity, clean accuracy decreased by 1.01 percentage points compared with the unpruned control.
- PGD-10 accuracy decreased by 0.42 percentage points between the control and the 70% pruned model.
- ASR remained close to 40%, without a consistent trend as sparsity increased.
- In this experiment, the pruned models retained most of the control model's measured clean and PGD-10 performance.
- The results do not demonstrate that pruning improves adversarial robustness.

**Artifacts**
- Pruning records: `experiments/adv_pruning_comparison.csv`.
- Evaluation results: `experiments/adv_pruning_evaluation.csv`.
- Figure: `experiments/adv_pruning_accuracy.png`.
- Checkpoints: `checkpoints/simple_cnn_adv_pruned_30.pth`, `checkpoints/simple_cnn_adv_pruned_50.pth`, and `checkpoints/simple_cnn_adv_pruned_70.pth`.

**Interpretation and limitations**

These preliminary results suggest that this FGSM adversarially trained SimpleCNN tolerated up to 70% global unstructured weight sparsity with relatively small changes in clean and PGD-10 accuracy. The experiment used one training checkpoint, one random seed, one dataset, and one PGD configuration. The pruned models were not fine-tuned. This experiment did not establish practical reductions in checkpoint storage, inference latency, or memory use. Results should not be generalized beyond the tested settings.

**Next steps**
1. Preserve these immediate post-pruning results as the baseline for a separate fine-tuning experiment.
2. Compare pruned models before and after fine-tuning.
3. Repeat training and pruning across multiple random seeds.
4. Evaluate additional attack settings under a consistent protocol.
5. Measure checkpoint size, inference latency, and memory usage separately from weight sparsity.

### Adversarially Trained SimpleCNN: Pruning and Fine-Tuning Comparison - 2026-11-01

**Status:** Fine-tuning and before-and-after evaluation completed; results are preliminary.

**Objective**

Determine whether short FGSM adversarial fine-tuning can recover clean and adversarial accuracy after global magnitude pruning of the FGSM adversarially trained SimpleCNN.

**Fine-tuning configuration**
- Dataset: CIFAR-10 training set, split into 45,000 training images and 5,000 validation images.
- Test set: CIFAR-10 test set (10,000 images), reserved for final comparison.
- Models: 30%, 50%, and 70% pruned adversarially trained SimpleCNN checkpoints.
- Fine-tuning method: FGSM adversarial training.
- Perturbation budget: epsilon = 8/255.
- Optimizer: Adam.
- Learning rate: 0.0001.
- Batch size: 64.
- Fine-tuning duration: 5 epochs.
- Random seed for the data split: 42.
- Original pruning masks were recovered from zero-valued weights. Gradient masking and post-optimizer weight masking were used to preserve the pruned connections.
- All three fine-tuned checkpoints completed training, and the original pruning zeros were preserved.
- Script: `finetune_pruned_models.py`.
- Validation history: `experiments/pruned_finetuning_history.csv`.

**Final evaluation configuration**
- Dataset: CIFAR-10 test set (10,000 images).
- Models: unpruned control, each pruned model, and each corresponding fine-tuned model.
- Attack: untargeted PGD-10 with random start.
- Perturbation budget: epsilon = 8/255.
- Step size: alpha = 2/255.
- Steps: 10.
- The same evaluation protocol was used for all seven models.
- Results: `experiments/adv_pruning_finetuning_comparison.csv`.
- Evaluation script: `eval_adv_pruning_finetune_comparison.py`.
- Plotting script: `plot_adv_pruning_finetuning.py`.

**Final test-set results**

| Model | Sparsity | Clean accuracy | PGD-10 accuracy | ASR |
|---|---:|---:|---:|---:|
| Control | 0% | 45.88% | 27.66% | 40.26% |
| Pruned | 30% | 45.47% | 27.55% | 39.78% |
| Fine-tuned | 30% | 47.32% | 28.30% | 40.43% |
| Pruned | 50% | 45.30% | 27.44% | 39.78% |
| Fine-tuned | 50% | 47.59% | 28.37% | 40.62% |
| Pruned | 70% | 44.87% | 27.24% | 39.98% |
| Fine-tuned | 70% | 47.40% | 27.97% | 41.18% |

ASR is calculated among test images that were correctly classified before the attack. PGD-10 accuracy is calculated over the full test set.

**Changes after fine-tuning**

| Sparsity | Clean accuracy change | PGD-10 accuracy change | ASR change |
|---|---:|---:|---:|
| 30% | +1.85 pp | +0.75 pp | +0.65 pp |
| 50% | +2.29 pp | +0.93 pp | +0.84 pp |
| 70% | +2.53 pp | +0.73 pp | +1.20 pp |

**Main observations**
- Fine-tuning increased clean accuracy at all three pruning levels.
- PGD-10 accuracy also increased slightly at all three pruning levels.
- ASR increased slightly after fine-tuning at every pruning level. This does not contradict the PGD-10 accuracy increase because the two metrics use different denominators.
- The 50% pruned and fine-tuned model had the highest clean accuracy (47.59%) and PGD-10 accuracy (28.37%) among the tested pruned configurations.
- The 70% pruned and fine-tuned model retained competitive accuracy, but had the highest ASR (41.18%) among the three fine-tuned models.

**Figures**
- `experiments/clean_accuracy_before_after.png`.
- `experiments/pgd10_accuracy_before_after.png`.
- `experiments/asr_before_after.png`.

All three figures were generated and verified on GitHub. Their plotted values match the comparison CSV.

**Interpretation and limitations**

For this single FGSM adversarially trained SimpleCNN checkpoint, five epochs of mask-preserving FGSM fine-tuning recovered clean accuracy and produced a small increase in PGD-10 accuracy after pruning. However, ASR also increased slightly. These results therefore do not establish an overall improvement in adversarial robustness. The experiment uses one training seed, one architecture, one dataset, and one PGD-10 configuration. Multi-restart and stronger attack evaluation, repeated training seeds, and measurements of checkpoint size, inference latency, and memory use are still needed.

**Next steps**
1. Repeat the pruning and fine-tuning experiment with multiple independent training seeds.
2. Evaluate the models with stronger PGD settings and multiple random restarts.
3. Measure checkpoint size, inference latency, and memory usage.
4. Compare the results with a standard architecture, prioritizing the CIFAR-10 ResNet-18 research track.

## ResNet-18: FGSM Adversarial Fine-Tuning and Evaluation - 2026-11-01

### Objective
Investigate whether FGSM adversarial fine-tuning improves the adversarial robustness of the CIFAR-10 ResNet-18 baseline.

### Training configuration
- Architecture: CIFAR-10 adapted ResNet-18
- Starting checkpoint: `checkpoints/resnet18_cifar10_best.pth`
- Dataset: CIFAR-10
- Training/validation split: 45,000 / 5,000 images
- Test set: 10,000 images, held out for final evaluation
- Attack used during training: FGSM
- Perturbation budget: epsilon = 8/255
- Batch size: 32
- Epochs: 10
- Optimizer: SGD
- Initial learning rate: 0.001
- Final learning rate: 0.00001
- Momentum: 0.9
- Weight decay: 0.0005
- Random seed: 42
- Best checkpoint selected using validation clean accuracy

### Training results
- Best validation clean accuracy: 77.24%
- Final test clean accuracy: 73.50%
- Final epoch adversarial training accuracy: 50.33%
- Final epoch training loss: 1.2858

### Baseline vs FGSM fine-tuned ResNet-18

| Evaluation | Baseline | FGSM fine-tuned |
|---|---:|---:|
| Clean accuracy | 89.26% | 73.50% |
| FGSM 2/255 | 22.46% | 65.27% |
| FGSM 4/255 | 10.31% | 57.36% |
| FGSM 8/255 | 5.63% | 42.15% |
| PGD-10 8/255 | 0.00% | 36.73% |

### Attack success rate

| Attack | Baseline ASR | FGSM fine-tuned ASR |
|---|---:|---:|
| FGSM 2/255 | 74.84% | 11.20% |
| FGSM 4/255 | 88.46% | 21.97% |
| FGSM 8/255 | 93.73% | 42.65% |
| PGD-10 8/255 | 100.00% | 50.03% |

### Interpretation
FGSM adversarial fine-tuning substantially improved adversarial accuracy under all tested attack settings. PGD-10 accuracy increased from 0.00% to 36.73%, while its attack success rate decreased from 100.00% to 50.03%.

However, clean accuracy decreased by 15.76 percentage points, from 89.26% to 73.50%. This indicates a trade-off between clean accuracy and adversarial robustness.

These results provide preliminary evidence that FGSM adversarial fine-tuning improves robustness, including against the tested iterative PGD attack. Stronger attacks and multiple random restarts are still needed.

### Limitations
- Only one random seed and one model checkpoint were evaluated.
- Only PGD-10 with one random start was tested.
- No multi-restart or stronger iterative attack evaluation has been conducted.
- The original baseline checkpoint was selected using test accuracy during training, whereas the fine-tuned checkpoint was selected using validation accuracy. This difference must be considered when interpreting the comparison.
- The experiment does not establish robustness against adaptive or other attack types.

### Saved artifacts
- `train_resnet18_adversarial.py`
- `eval_resnet18_fgsm_adv_comparison.py`
- `checkpoints/resnet18_cifar10_fgsm_adv.pth`
- `experiments/resnet18_fgsm_adv_history.csv`
- `experiments/resnet18_fgsm_adv_result.txt`
- `experiments/resnet18_fgsm_adv_comparison.csv`

### Status
FGSM adversarial fine-tuning and baseline comparison completed. Stronger robustness evaluation remains.

## ResNet-18: PGD-50 Evaluation with Five Random Restarts - 2026-11-02

### Configuration
- Dataset: CIFAR-10 test set (10,000 images)
- Models: baseline ResNet-18 and FGSM adversarially trained ResNet-18
- Attack: untargeted PGD under the L-infinity norm
- Epsilon: 8/255
- Step size: 2/255
- PGD steps: 50
- Random restarts: 5
- Batch size: 32
- Base seed: 42
- Device: NVIDIA GeForce GTX 1050

### Results

| Model | Clean Accuracy | PGD-50 Robust Accuracy | Cumulative ASR |
|---|---:|---:|---:|
| Baseline ResNet-18 | 89.26% | 0.00% | 100.00% |
| FGSM Adversarial ResNet-18 | 73.50% | 34.97% | 52.42% |

### Restart Results

| Model | Successful Attacks per Restart |
|---|---|
| Baseline | 8926, 8926, 8926, 8926, 8926 |
| FGSM Adversarial | 3815, 3821, 3814, 3820, 3820 |

Cumulative ASR combines successful attacks across all five restarts using a per-image logical OR. Each initially correctly classified image is counted at most once.

### Observations
- The baseline model had 0.00% robust accuracy under this PGD-50 evaluation.
- FGSM adversarial training improved PGD-50 robust accuracy to 34.97%.
- Clean accuracy decreased by 15.76 percentage points compared with the baseline.
- The FGSM adversarial model had a cumulative ASR of 52.42%.
- The five restarts produced similar individual attack success counts.

### Limitations
- Results use one trained checkpoint per model and one base seed.
- Only one epsilon and one PGD step size were evaluated.
- The results do not establish robustness against all possible attacks.
- The baseline checkpoint was selected using test accuracy during training, so its clean accuracy is subject to test-set selection bias.
- Further evaluation with stronger attacks and multiple training seeds is needed.

### Files
- Script: `eval_resnet18_pgd50_multirestart.py`
- Results: `experiments/resnet18_pgd50_multirestart.csv`


## ResNet-18: Magnitude Pruning of FGSM Adversarially Trained Model - 2026-11-02

### Objective
Investigate how global unstructured magnitude pruning affects the clean accuracy and adversarial robustness of an FGSM adversarially trained CIFAR-10 ResNet-18.

### Configuration
- Dataset: CIFAR-10 test set (10,000 images)
- Starting checkpoint: `checkpoints/resnet18_cifar10_fgsm_adv.pth`
- Architecture: CIFAR-adapted ResNet-18
- Pruning method: Global unstructured L1 magnitude pruning
- Prunable layers: Conv2d and Linear weights
- Pruning ratios: 30%, 50%, 70%
- Models: Original control and three independently pruned models
- Device: NVIDIA GeForce GTX 1050
- Evaluation batch size: 64
- Random seed: 42

### Verified sparsity

| Model | Target sparsity | Actual sparsity |
|---|---:|---:|
| Control | 0% | 0% |
| Pruned 30% | 30% | 30.00% |
| Pruned 50% | 50% | 50.00% |
| Pruned 70% | 70% | 70.00% |

The original checkpoint was independently loaded for each pruning ratio. Pruning masks were removed before saving, leaving zero-valued weights in dense tensors.

### Evaluation configuration
- FGSM epsilon: 2/255, 4/255, 8/255
- PGD epsilon: 8/255
- PGD step size: 2/255
- PGD steps: 10
- PGD random start: Enabled
- Metrics: Clean accuracy, adversarial accuracy, attack success rate (ASR)

### Clean and adversarial accuracy

| Model | Sparsity | Clean | FGSM 2/255 | FGSM 4/255 | FGSM 8/255 | PGD-10 8/255 |
|---|---:|---:|---:|---:|---:|---:|
| Control | 0% | 73.50% | 65.27% | 57.36% | 42.15% | 36.73% |
| Pruned 30% | 30% | 73.44% | 65.48% | 57.33% | 42.23% | 36.69% |
| Pruned 50% | 50% | 72.43% | 64.85% | 57.07% | 42.01% | 36.99% |
| Pruned 70% | 70% | 61.50% | 53.70% | 46.81% | 33.80% | 30.57% |

### Attack success rate

| Model | FGSM 2/255 | FGSM 4/255 | FGSM 8/255 | PGD-10 8/255 |
|---|---:|---:|---:|---:|
| Control | 11.20% | 21.97% | 42.65% | 50.03% |
| Pruned 30% | 10.85% | 21.95% | 42.50% | 50.04% |
| Pruned 50% | 10.48% | 21.23% | 42.01% | 48.94% |
| Pruned 70% | 12.70% | 23.92% | 45.06% | 50.31% |

ASR is calculated among samples that were correctly classified before the attack.

### Observations
- 30% pruning had almost no effect on clean or adversarial accuracy compared with the control.
- 50% pruning caused a 1.07 percentage-point decrease in clean accuracy. PGD-10 accuracy was slightly higher than the control, but the difference was small.
- 70% pruning reduced clean accuracy by 12.00 percentage points and PGD-10 accuracy by 6.16 percentage points.
- The results do not show a consistent improvement in adversarial robustness as sparsity increases.
- The 70% pruned model had the highest PGD-10 ASR at 50.31%, while the 50% model had the lowest at 48.94%.

### Limitations
- Only one trained checkpoint and one random seed were evaluated.
- Only one PGD configuration was tested: 10 steps, epsilon 8/255, and step size 2/255.
- No multi-restart PGD evaluation was conducted for these pruned models.
- The pruned models contain dense tensors with zero-valued weights. Their checkpoint file sizes remained unchanged, so actual storage or inference-speed improvements have not been demonstrated.
- No fine-tuning after ResNet-18 pruning was conducted in this experiment.
- Results do not establish robustness against adaptive or other attack types.

### Saved artifacts
- Pruning script: `prune_resnet18_adv.py`
- Evaluation script: `eval_resnet18_adv_pruning.py`
- Sparsity results: `experiments/resnet18_adv_pruning_sparsity.csv`
- Accuracy and ASR results: `experiments/resnet18_adv_pruning_comparison.csv`
- Control checkpoint: `checkpoints/resnet18_cifar10_fgsm_adv.pth`
- Pruned checkpoints:
  - `checkpoints/resnet18_cifar10_fgsm_adv_pruned_30.pth`
  - `checkpoints/resnet18_cifar10_fgsm_adv_pruned_50.pth`
  - `checkpoints/resnet18_cifar10_fgsm_adv_pruned_70.pth`

### Status
ResNet-18 magnitude pruning and FGSM/PGD-10 evaluation completed. Further work is needed to investigate fine-tuning after pruning, multi-restart robustness, and actual model compression or inference efficiency.

## ResNet-18: Fine-Tuning After Magnitude Pruning - 2026-11-03

**Status:** Fine-tuning and comparative evaluation completed; results are preliminary.

### Objective

Investigate whether FGSM adversarial fine-tuning can recover clean accuracy and adversarial performance after global magnitude pruning of an FGSM adversarially trained CIFAR-10 ResNet-18.

### Fine-tuning configuration

- Dataset: CIFAR-10.
- Training/validation split: 45,000 / 5,000 images.
- Test set: 10,000 images, reserved for final evaluation.
- Starting checkpoints: ResNet-18 models pruned at 30%, 50%, and 70%.
- Fine-tuning method: FGSM adversarial training.
- Perturbation budget: epsilon = 8/255.
- Batch size: 32.
- Epochs: 10.
- Optimizer: SGD.
- Initial learning rate: 0.001.
- Momentum: 0.9.
- Weight decay: 0.0005.
- Learning-rate scheduler: MultiStepLR, milestones at epochs 6 and 8, gamma = 0.1.
- Random seed: 42.
- Pruning masks were reconstructed from zero-valued weights. Gradients were masked, and pruning zeros were restored after optimizer updates.
- Best checkpoints were selected using validation clean accuracy.

### Training results

| Pruning ratio | Best validation accuracy | Final measured sparsity |
|---|---:|---:|
| 30% | 78.86% | 30.00% |
| 50% | 78.96% | 50.00% |
| 70% | 78.70% | 70.00% |

### Evaluation configuration

- Dataset: CIFAR-10 test set (10,000 images).
- Models: unpruned control, three pruned models, and three fine-tuned models.
- FGSM epsilon: 2/255, 4/255, and 8/255.
- PGD epsilon: 8/255.
- PGD step size: 2/255.
- PGD steps: 10.
- PGD random start: enabled.
- PGD restarts: one.
- Metrics: clean accuracy, adversarial accuracy, and attack success rate (ASR).
- ASR was calculated among samples correctly classified before the attack.

### Clean and adversarial accuracy

| Model | Sparsity | Clean | FGSM 2/255 | FGSM 4/255 | FGSM 8/255 | PGD-10 |
|---|---:|---:|---:|---:|---:|---:|
| Control | 0% | 73.50% | 65.27% | 57.36% | 42.15% | 36.73% |
| Pruned 30% | 30% | 73.44% | 65.48% | 57.33% | 42.23% | 36.69% |
| Pruned 50% | 50% | 72.43% | 64.85% | 57.07% | 42.01% | 36.99% |
| Pruned 70% | 70% | 61.50% | 53.70% | 46.81% | 33.80% | 30.57% |
| Fine-tuned 30% | 30% | 75.45% | 66.06% | 56.60% | 40.98% | 33.71% |
| Fine-tuned 50% | 50% | 75.48% | 66.40% | 57.12% | 41.27% | 33.90% |
| Fine-tuned 70% | 70% | 75.22% | 66.57% | 57.78% | 42.02% | 35.37% |

### Attack success rate

| Model | FGSM 2/255 | FGSM 4/255 | FGSM 8/255 | PGD-10 |
|---|---:|---:|---:|---:|
| Control | 11.20% | 21.97% | 42.65% | 50.03% |
| Pruned 30% | 10.85% | 21.95% | 42.50% | 50.04% |
| Pruned 50% | 10.48% | 21.23% | 42.01% | 48.94% |
| Pruned 70% | 12.70% | 23.92% | 45.06% | 50.31% |
| Fine-tuned 30% | 12.45% | 24.98% | 45.69% | 55.32% |
| Fine-tuned 50% | 12.03% | 24.32% | 45.32% | 55.09% |
| Fine-tuned 70% | 11.50% | 23.19% | 44.14% | 52.98% |

### Changes after fine-tuning

| Sparsity | Clean accuracy change | FGSM 8/255 change | PGD-10 change |
|---|---:|---:|---:|
| 30% | +2.01 pp | -1.25 pp | -2.98 pp |
| 50% | +3.05 pp | -0.74 pp | -3.09 pp |
| 70% | +13.72 pp | +8.22 pp | +4.80 pp |

### Main observations

- Fine-tuning increased clean accuracy at all three pruning levels.
- The 70% pruned model recovered substantially, increasing clean accuracy from 61.50% to 75.22%.
- For 30% and 50% pruning, clean accuracy improved, but PGD-10 accuracy decreased after fine-tuning.
- At 70% pruning, both clean and adversarial accuracy improved compared with the model before fine-tuning.
- ASR increased after fine-tuning for all three pruning levels under PGD-10. This is not contradictory to the accuracy results because ASR is calculated only among initially correctly classified samples, whereas adversarial accuracy uses the full test set.
- Compared with the unpruned control, all three fine-tuned models achieved higher clean accuracy but lower PGD-10 accuracy.

### Interpretation and limitations

Fine-tuning helped recover clean accuracy after pruning, especially at 70% sparsity. However, the effect on adversarial robustness was not consistent across pruning levels. The 30% and 50% models lost PGD-10 accuracy after fine-tuning, while the 70% model improved relative to its pre-fine-tuning result.

These findings do not establish that fine-tuning improves adversarial robustness overall. The evaluation used one training seed, one checkpoint per model, and PGD-10 with one random start. The pruning masks were reconstructed from zero-valued weights rather than restored from separately saved mask metadata.

The pruned checkpoints contain dense tensors with zero-valued weights. Their file sizes remained approximately 44.78 MB, so practical storage or inference-speed improvements have not been demonstrated.

### Saved artifacts

- Fine-tuning script: `train_resnet18_pruned_finetuning.py`
- Evaluation script: `eval_resnet18_pruned_finetuning.py`
- Evaluation results: `experiments/resnet18_pruned_finetuning_comparison.csv`
- Training summary: `experiments/resnet18_pruned_finetuning_summary.csv`
- Training histories:
  - `experiments/resnet18_pruned_30_finetuning_history.csv`
  - `experiments/resnet18_pruned_50_finetuning_history.csv`
  - `experiments/resnet18_pruned_70_finetuning_history.csv`
- Fine-tuned checkpoints:
  - `checkpoints/resnet18_cifar10_fgsm_adv_pruned_30_finetuned.pth`
  - `checkpoints/resnet18_cifar10_fgsm_adv_pruned_50_finetuned.pth`
  - `checkpoints/resnet18_cifar10_fgsm_adv_pruned_70_finetuned.pth`

### Next steps

1. Evaluate the fine-tuned models using PGD-50 with five random restarts.
2. Compare the results against the existing unpruned ResNet-18 PGD-50 evaluation.
3. Repeat the pruning and fine-tuning experiments across multiple independent training seeds.
4. Measure checkpoint storage, inference latency, and memory usage using suitable sparse implementations.
5. Investigate whether quantization can be combined with pruning and adversarial fine-tuning.

## ResNet-18: PGD-50 Multi-Restart Evaluation of Fine-Tuned Pruned Models - 2026-11-03

### Objective

Evaluate the adversarial robustness of the three fine-tuned pruned ResNet-18 models using a stronger PGD attack with 50 steps and five random restarts. Compare them with the original baseline and the unpruned FGSM-adversarially-trained control.

### Experimental Configuration

- Dataset: CIFAR-10 test set (10,000 images)
- Model: CIFAR-adapted ResNet-18
- Batch size: 32
- Attack: Untargeted PGD
- Epsilon: 8/255
- Step size: 2/255
- PGD steps: 50
- Random restarts: 5
- Base seed: 42
- Device: NVIDIA GeForce GTX 1050
- Evaluation: Clean accuracy, PGD-50 robust accuracy, cumulative attack success rate (ASR)

Cumulative ASR counts an image as successfully attacked if at least one of the five restarts causes misclassification, among images correctly classified before the attack. Robust accuracy is calculated over all 10,000 test images.

### Results

| Model | Clean Accuracy | PGD-50 Robust Accuracy | Cumulative ASR |
|---|---:|---:|---:|
| Baseline | 89.26% | 0.00% | 100.00% |
| FGSM Adversarial Control | 73.50% | 34.97% | 52.42% |
| Fine-tuned 30% | 75.45% | 31.62% | 58.09% |
| Fine-tuned 50% | 75.48% | 31.81% | 57.86% |
| Fine-tuned 70% | 75.22% | 33.45% | 55.53% |

### Observations

1. The unpruned FGSM-adversarially-trained control achieved the highest PGD-50 robust accuracy at 34.97%.
2. Among the fine-tuned pruned models, the 70% model achieved the highest PGD-50 robust accuracy at 33.45%.
3. The 30% and 50% fine-tuned models achieved slightly higher clean accuracy than the control, but their PGD-50 robust accuracy was lower.
4. The 70% fine-tuned model recovered much of the performance lost after pruning. Its PGD-50 robust accuracy reached 33.45%, compared with 30.57% for the corresponding pruned model before fine-tuning under PGD-10. These values use different attack configurations, so they should not be interpreted as a direct controlled comparison.
5. None of the fine-tuned pruned models surpassed the unpruned adversarially-trained control under this PGD-50 evaluation.
6. Increasing the pruning ratio did not produce a consistent improvement in adversarial robustness.

### Limitations

- Only one training seed and one checkpoint per model were evaluated.
- PGD-50 used five random restarts, but this does not establish formal worst-case robustness.
- The results are specific to CIFAR-10, this architecture, and the selected attack configuration.
- The experiment does not demonstrate actual storage reduction or inference-speed improvement from unstructured pruning.

### Artifacts

- Evaluation script: `eval_resnet18_pruned_finetuned_pgd50.py`
- Results: `experiments/resnet18_pruned_finetuned_pgd50.csv`

### Next Steps

1. Repeat experiments with multiple training seeds.
2. Investigate why fine-tuning after 70% pruning recovers more adversarial performance than at 30% and 50%.
3. Evaluate actual inference latency, memory consumption, and model storage using an appropriate sparse implementation.
4. Investigate quantization and combined pruning-quantization approaches.

## ResNet-18: Inference Efficiency Benchmark - 2026-11-04

### Objective

Measure whether unstructured magnitude pruning and fine-tuning reduce the practical inference cost of ResNet-18 on the NVIDIA GTX 1050.

The final controlled comparison evaluated the original baseline and the 70%-pruned, adversarially fine-tuned model.

### Final Benchmark Configuration

- GPU: NVIDIA GeForce GTX 1050 (4 GB)
- Device: CUDA
- Batch size: 64
- Warm-up iterations: 50 per measurement
- Timed iterations: 100 per measurement
- Repeated measurements: 10 per model
- Measurement order: randomized between models
- Timing method: PyTorch CUDA events with synchronization
- Models: Baseline and Fine-tuned 70%
- Input: Random tensor with shape `[64, 3, 32, 32]`
- Execution mode: `torch.inference_mode()`

Both models were loaded onto the GPU before benchmarking.

### Final Controlled Results

| Metric | Baseline | Fine-tuned 70% |
|---|---:|---:|
| Median latency (ms) | 49.8516 | 49.7795 |
| Mean latency (ms) | 50.6292 | 49.7716 |
| Standard deviation (ms) | 2.7775 | 0.2514 |
| Throughput (images/s) | 1283.81 | 1285.67 |
| Checkpoint size (MiB) | 42.7021 | 42.7051 |

### Raw Measurement Inspection

The Baseline model had one unusually slow measurement in round 10:

- Baseline: 58.5055 ms
- Fine-tuned 70%: 50.0989 ms

The other measurements were generally close to 50 ms.

This outlier increased the Baseline standard deviation. The median is therefore more representative than the mean for this comparison.

### Observations and Interpretation

1. The median latency difference was approximately 0.0721 ms, or 0.14%. This is too small to establish a meaningful inference-speed improvement.
2. Throughput was almost identical between the two models.
3. The Fine-tuned 70% model had lower timing variation in this particular run. The reason for the Baseline outlier was not established.
4. Checkpoint sizes remained approximately 42.7 MiB. Unstructured pruning did not produce meaningful dense checkpoint storage reduction.
5. Although the model achieved approximately 70% sparsity among eligible weights, this sparsity did not translate into measurable practical inference acceleration in this experiment.

### Earlier Benchmark Attempts

Several earlier benchmark versions were conducted to investigate timing variability:

- `benchmark_resnet18_efficiency.py`: initial benchmark.
- `benchmark_resnet18_efficiency_controlled.py`: randomized batch-size measurements.
- `benchmark_resnet18_efficiency_v2.py`: randomized model and batch-size measurements.
- `benchmark_resnet18_efficiency_v3.py`: 10 randomized rounds with raw measurements.
- `benchmark_resnet18_efficiency_final.py`: focused Baseline versus Fine-tuned 70% comparison.

The earlier measurements showed substantial timing variability and are retained as diagnostic experiments. The final focused comparison is reported separately and does not replace those results.

### Limitations

- Only one GPU was used.
- Only one checkpoint per model was evaluated.
- Only batch size 64 was included in the final focused comparison.
- Ten measurements per model are insufficient to establish a general performance difference.
- The Baseline had one timing outlier, and its cause could not be determined.
- Random input tensors were used rather than the CIFAR-10 test dataset.
- No sparse inference kernel or hardware-accelerated sparse implementation was used.
- Checkpoint size does not represent compressed sparse storage.
- GPU power measurements were unavailable through the NVIDIA driver interface.

### Saved Artifacts

- Final benchmark script: `benchmark_resnet18_efficiency_final.py`
- Raw measurements: `experiments/resnet18_efficiency_final_raw.csv`
- Summary results: `experiments/resnet18_efficiency_final_summary.csv`
- Earlier diagnostic scripts and results are retained separately.

### Status

The preliminary inference-efficiency comparison is complete.

The current evidence does not demonstrate a meaningful inference-speed or storage improvement from 70% unstructured pruning on the GTX 1050. Further work would require a sparse inference implementation and more repeated measurements under controlled conditions.
### ResNet-18 Validation-Based Baseline - 2026-11-05

**Status:** Completed; single-seed baseline and final test evaluation.

**Dataset and split**
- Dataset: CIFAR-10.
- Training subset: 45,000 images.
- Validation subset: 5,000 images.
- Test set: 10,000 images, reserved for final evaluation.
- Random seed: 42.
- Input transformation: ToTensor(), pixel range [0, 1].

**Training configuration**
- Architecture: CIFAR-adapted ResNet-18.
- Parameters: 11,173,962.
- Epochs: 100.
- Batch size: 64.
- Optimizer: SGD.
- Learning rate: 0.1.
- Momentum: 0.9.
- Weight decay: 0.0005.
- Scheduler: MultiStepLR, milestones [50, 75], gamma 0.1.
- Best checkpoint selected using validation accuracy only.

**Training results**
- Best validation accuracy: 89.70% at epoch 54.
- Final training accuracy: 100.00%.
- Final validation accuracy: 89.36%.
- Final validation loss: 0.3746.

**Final test results**

Clean accuracy: **88.06%**.

| Attack | Epsilon | Adversarial accuracy | ASR |
|---|---:|---:|---:|
| FGSM | 2/255 | 21.39% | 75.71% |
| FGSM | 4/255 | 8.68% | 90.14% |
| FGSM | 8/255 | 3.85% | 95.70% |
| PGD-10 | 8/255 | 0.00% | 100.00% |

PGD configuration: 10 steps, alpha 2/255, random start.

**Artifacts**
- Training: `train_resnet18_validated.py`
- Evaluation: `eval_resnet18_validated.py`
- Best checkpoint: `checkpoints/resnet18_cifar10_validated_best.pth`
- Latest checkpoint: `checkpoints/resnet18_cifar10_validated.pth`
- Training history: `experiments/resnet18_validated_training_history.csv`
- Attack results: `experiments/resnet18_validated_attack_results.csv`

**Limitations**
- Only one training seed was used.
- ASR code and reported percentages were reviewed, but an independent per-example audit remains unfinished.
- Existing ResNet-18 adversarial-training and pruning experiments still use the previous test-selected baseline.
- Those downstream experiments must be repeated using this new baseline before making direct comparisons.

**Conclusion**

This validation-based baseline provides a more methodologically appropriate starting point for subsequent controlled ResNet-18 adversarial-training and pruning experiments.


---

# Research Updates: Validation-Based Experiments (2026-11-05 to 2026-11-08)

This section records experiments completed after the validation-based ResNet-18 baseline was established. All earlier sections and historical results are retained unchanged.

## 1. Independent ASR Audit - 2026-11-06

### Objective

Independently verify the attack success rate (ASR) calculations for the validation-based ResNet-18 baseline.

### Method

- Model: Validation-based CIFAR-10 ResNet-18.
- Test set: 10,000 images.
- Clean-correct samples: 8,806.
- ASR calculated conditionally on samples correctly classified before the attack.
- FGSM and PGD attack results independently recalculated.

### Results

| Attack | Epsilon | Clean Accuracy | Adversarial Accuracy | ASR | Successful Attacks |
|---|---:|---:|---:|---:|---:|
| FGSM | 2/255 | 88.06% | 21.39% | 75.71% | 6,667 / 8,806 |
| FGSM | 4/255 | 88.06% | 8.68% | 90.14% | 7,938 / 8,806 |
| FGSM | 8/255 | 88.06% | 3.85% | 95.70% | 8,427 / 8,806 |
| PGD-10 | 8/255 | 88.06% | 0.00% | 100.00% | 8,806 / 8,806 |

### Conclusion

The independent audit reproduced the original ASR results. The reported ASR calculations for the validation-based baseline are consistent with the independently recalculated per-example results.

### Artifacts

- Audit script: `audit_resnet18_validated_asr.py`
- Audit results: `experiments/resnet18_validated_asr_audit.csv`

## 2. Validation-Based FGSM Adversarial Fine-Tuning - 2026-11-06

### Objective

Investigate whether FGSM adversarial fine-tuning improves the adversarial robustness of the validation-based ResNet-18 baseline.

### Experimental Configuration

- Dataset: CIFAR-10.
- Training images: 45,000.
- Validation images: 5,000.
- Test images: 10,000.
- Starting checkpoint: `checkpoints/resnet18_cifar10_validated_best.pth`
- Architecture: CIFAR-adapted ResNet-18.
- Training epochs: 10.
- Batch size: 32.
- Attack: FGSM.
- Epsilon: 8/255.
- Optimizer: SGD.
- Learning rate: 0.001.
- Momentum: 0.9.
- Weight decay: 0.0005.
- Scheduler milestones: 6 and 8.
- Scheduler gamma: 0.1.

The best checkpoint was selected using validation accuracy.

### Results

- Best validation accuracy: 72.64% at epoch 8.
- Final test clean accuracy: 68.93%.
- Final training adversarial accuracy: 40.55%.
- Original validation-based baseline clean test accuracy: 88.06%.
- Clean accuracy change: -19.13 percentage points.

### Interpretation

FGSM adversarial fine-tuning substantially changes the clean-accuracy and robustness characteristics of the original model. The clean-accuracy reduction must be considered when evaluating the robustness improvements.

### Artifacts

- Training script: `train_resnet18_validated_fgsm_adv.py`
- Best checkpoint: `checkpoints/resnet18_validated_fgsm_adv_best.pth`
- Training history: `experiments/resnet18_validated_fgsm_adv_history.csv`
- Training result: `experiments/resnet18_validated_fgsm_adv_result.txt`

## 3. FGSM and PGD-10 Comparison - 2026-11-07

### Objective

Compare the validation-based standard baseline with its FGSM-adversarially-trained counterpart under clean and adversarial evaluation.

### Results: Clean and Adversarial Accuracy

| Evaluation | Validated Baseline | FGSM-Trained | Change |
|---|---:|---:|---:|
| Clean accuracy | 88.06% | 68.93% | -19.13 pp |
| FGSM 2/255 | 21.39% | 60.97% | +39.58 pp |
| FGSM 4/255 | 8.68% | 52.67% | +43.99 pp |
| FGSM 8/255 | 3.85% | 38.09% | +34.24 pp |
| PGD-10 8/255 | 0.00% | 33.49% | +33.49 pp |

### Results: Attack Success Rate

| Attack | Validated Baseline | FGSM-Trained | ASR Reduction |
|---|---:|---:|---:|
| FGSM 2/255 | 75.71% | 11.55% | 64.16 pp |
| FGSM 4/255 | 90.14% | 23.59% | 66.55 pp |
| FGSM 8/255 | 95.70% | 44.74% | 50.96 pp |
| PGD-10 8/255 | 100.00% | 51.41% | 48.59 pp |

### Interpretation

FGSM adversarial fine-tuning improved adversarial accuracy and reduced ASR under all evaluated attacks.

However, the improvement came with a 19.13 percentage-point reduction in clean accuracy. The results indicate a substantial robustness-versus-clean-accuracy trade-off.

These results alone do not establish robust generalization across different training seeds or attack configurations.

### Artifacts

- Comparison script: `eval_resnet18_validated_fgsm_adv_comparison.py`
- Results: `experiments/resnet18_validated_fgsm_adv_comparison.csv`

## 4. PGD-50 Multi-Restart Evaluation - 2026-11-07

### Objective

Evaluate whether the robustness improvement from FGSM adversarial fine-tuning remains under a stronger iterative attack.

### Experimental Configuration

- Dataset: CIFAR-10 test set (10,000 images).
- Attack: Untargeted PGD.
- Epsilon: 8/255.
- Step size: 2/255.
- PGD steps: 50.
- Random restarts: 5.
- Evaluation models:
  - Validation-based standard baseline.
  - Validation-based FGSM-adversarially-trained model.

Cumulative ASR counts an initially correctly classified image as successfully attacked if at least one of the five restarts causes misclassification.

### Results

| Metric | Validated Baseline | FGSM-Trained |
|---|---:|---:|
| Clean accuracy | 88.06% | 68.93% |
| PGD-50 robust accuracy | 0.00% | 32.25% |
| Cumulative PGD-50 ASR | 100.00% | 53.21% |

### Interpretation

The FGSM-trained model retained a measurable robustness improvement under PGD-50 with five random restarts.

Its PGD-50 robust accuracy was 32.25%, compared with 0.00% for the standard baseline. Nevertheless, the FGSM-trained model remained vulnerable to the stronger attack.

The clean-accuracy reduction of 19.13 percentage points remains an important limitation.

### Artifacts

- Evaluation script: `eval_resnet18_validated_pgd50_multirestart.py`
- Results: `experiments/resnet18_validated_pgd50_multirestart.csv`

## 5. Validation-Based Standard Baseline Pruning - 2026-11-07

### Objective

Evaluate how global unstructured magnitude pruning affects the clean accuracy and adversarial robustness of the validation-based standard ResNet-18 baseline.

### Experimental Configuration

- Source checkpoint: `checkpoints/resnet18_cifar10_validated_best.pth`
- Dataset: CIFAR-10 test set (10,000 images).
- Pruning method: Global L1 unstructured magnitude pruning.
- Pruning ratios: 30%, 50%, and 70%.
- Total parameters: 11,173,962.
- Prunable weights: 11,164,352.
- Prunable layers: 21.
- Attacks: FGSM at 2/255, 4/255, and 8/255; PGD-10 at 8/255.
- PGD step size: 2/255.
- ASR calculated conditionally on initially correct samples.

### Clean Accuracy Results

| Model | Sparsity | Clean Accuracy | Change vs Control |
|---|---:|---:|---:|
| Control | 0% | 88.06% | Baseline |
| Pruned | 30% | 88.05% | -0.01 pp |
| Pruned | 50% | 87.91% | -0.15 pp |
| Pruned | 70% | 86.58% | -1.48 pp |

### Adversarial Accuracy Results

| Attack | Control | 30% | 50% | 70% |
|---|---:|---:|---:|---:|
| FGSM 2/255 | 21.39% | 21.25% | 20.70% | 18.70% |
| FGSM 4/255 | 8.68% | 8.60% | 7.97% | 5.93% |
| FGSM 8/255 | 3.85% | 3.88% | 3.44% | 2.51% |
| PGD-10 8/255 | 0.00% | 0.00% | 0.00% | 0.00% |

### ASR Results

| Attack | Control | 30% | 50% | 70% |
|---|---:|---:|---:|---:|
| FGSM 2/255 | 75.71% | 75.87% | 76.45% | 78.40% |
| FGSM 4/255 | 90.14% | 90.24% | 90.93% | 93.15% |
| FGSM 8/255 | 95.70% | 95.67% | 96.14% | 97.11% |
| PGD-10 8/255 | 100.00% | 100.00% | 100.00% | 100.00% |

### Interpretation

The standard baseline tolerated 30% and 50% unstructured pruning with very small clean-accuracy changes. At 70% pruning, clean and adversarial accuracy declined more noticeably.

Pruning did not demonstrate a consistent adversarial robustness improvement. All models achieved 0% robust accuracy under the evaluated PGD-10 attack.

### Artifacts

- Pruning script: `prune_resnet18_validated_baseline.py`
- Pruning sparsity results: `experiments/resnet18_validated_baseline_pruning_sparsity.csv`
- Evaluation script: `eval_resnet18_validated_baseline_pruning.py`
- Evaluation results: `experiments/resnet18_validated_baseline_pruning_evaluation.csv`

## 6. Validation-Based FGSM-Trained Model Pruning - 2026-11-08

### Objective

Investigate how global unstructured magnitude pruning affects the clean accuracy and adversarial robustness of the validation-based FGSM-adversarially-trained ResNet-18 model.

### Experimental Configuration

- Source checkpoint: `checkpoints/resnet18_validated_fgsm_adv_best.pth`
- Dataset: CIFAR-10 test set (10,000 images).
- Pruning method: Global L1 unstructured magnitude pruning.
- Pruning ratios: 30%, 50%, and 70%.
- Total parameters: 11,173,962.
- Prunable weights: 11,164,352.
- Prunable layers: 21.
- Attacks: FGSM at 2/255, 4/255, and 8/255; PGD-10 at 8/255.
- PGD step size: 2/255.
- ASR calculated conditionally on initially correct samples.

### Clean Accuracy Results

| Model | Sparsity | Clean Accuracy | Change vs Control |
|---|---:|---:|---:|
| FGSM-trained control | 0% | 68.93% | Baseline |
| Pruned | 30% | 68.79% | -0.14 pp |
| Pruned | 50% | 67.80% | -1.13 pp |
| Pruned | 70% | 54.54% | -14.39 pp |

### Adversarial Accuracy Results

| Attack | Control | 30% | 50% | 70% |
|---|---:|---:|---:|---:|
| FGSM 2/255 | 60.97% | 60.90% | 59.96% | 47.98% |
| FGSM 4/255 | 52.67% | 52.77% | 52.03% | 41.72% |
| FGSM 8/255 | 38.09% | 38.04% | 37.70% | 31.29% |
| PGD-10 8/255 | 33.49% | 33.69% | 33.61% | 28.94% |

### ASR Results

| Attack | Control | 30% | 50% | 70% |
|---|---:|---:|---:|---:|
| FGSM 2/255 | 11.55% | 11.48% | 11.59% | 12.06% |
| FGSM 4/255 | 23.59% | 23.29% | 23.30% | 23.56% |
| FGSM 8/255 | 44.74% | 44.70% | 44.42% | 42.74% |
| PGD-10 8/255 | 51.41% | 51.02% | 50.44% | 47.25% |

### Interpretation

The 30% and 50% pruning configurations preserved most of the FGSM-trained model's clean and adversarial accuracy.

The 70% pruning configuration caused a substantial clean-accuracy reduction of 14.39 percentage points and reduced absolute adversarial accuracy.

The lower ASR values observed for some heavily pruned models should not independently be interpreted as improved robustness. ASR is conditional on initially correct samples, so it must be considered alongside clean accuracy and adversarial accuracy.

The small differences between the control, 30%, and 50% models do not establish a meaningful robustness improvement from pruning.

### Artifacts

- Pruning script: `prune_resnet18_validated_fgsm_adv.py`
- Pruning sparsity results: `experiments/resnet18_validated_fgsm_adv_pruning_sparsity.csv`
- Evaluation script: `eval_resnet18_validated_fgsm_adv_pruning.py`
- Evaluation results: `experiments/resnet18_validated_fgsm_adv_pruning_evaluation.csv`

## 7. Combined Validation-Based Pruning Results - 2026-11-08

The standard-baseline and FGSM-trained-model pruning evaluation CSV files were combined into one dataset for easier analysis.

- Standard baseline evaluation: `experiments/resnet18_validated_baseline_pruning_evaluation.csv`
- FGSM-trained evaluation: `experiments/resnet18_validated_fgsm_adv_pruning_evaluation.csv`
- Combined evaluation: `experiments/resnet18_validated_combined_pruning_evaluation.csv`
- Total combined rows: 32.

The original evaluation CSV files were retained unchanged.

## 8. Overall Findings from the Validation-Based Experiments

1. The validation-based ResNet-18 baseline achieved 88.06% clean test accuracy.
2. The independent audit confirmed the baseline ASR calculations.
3. FGSM adversarial fine-tuning improved adversarial accuracy under FGSM, PGD-10, and PGD-50 evaluation.
4. The robustness improvement from FGSM fine-tuning came with a 19.13 percentage-point clean-accuracy reduction.
5. Standard-baseline pruning at 30% and 50% caused very small clean-accuracy changes but did not improve adversarial robustness.
6. Pruning the FGSM-trained model at 30% and 50% largely preserved its performance, while 70% pruning caused substantial degradation.
7. The current results do not establish that unstructured pruning provides practical inference acceleration or storage reduction.
8. All validation-based training and pruning comparisons currently rely on one training seed.

These findings should be interpreted in light of the single-training-seed limitation. Multi-seed validation is outside the current project scope and may be considered in future work.

