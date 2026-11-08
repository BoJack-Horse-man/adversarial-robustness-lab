# Research Plan: Adversarial Robustness of Efficient Neural Networks

## Goal

Investigate how adversarial training and model compression affect the clean accuracy and adversarial robustness of neural networks. The experimental work uses CIFAR-10 and CIFAR-100, beginning with SimpleCNN and extending to a CIFAR-adapted ResNet-18. HYDRA is a reference method for study; its full reproduction is not a completed project result.

## Working principles

- Work one milestone at a time and verify each step before moving on.
- Record commands, environment details, errors, fixes, experiment settings, and results.
- Keep our implementation separate from third-party repositories.
- Never claim an experiment is reproduced until the settings and results have been checked.
- Distinguish smoke tests from full reproductions.
- Report clean accuracy, adversarial accuracy, and ASR with their definitions and denominators.
- Keep train, validation, and test roles explicit. Do not use test-set results for model selection.
- Treat single-seed results and preliminary benchmarks as exploratory evidence, not general conclusions.
- Keep datasets, checkpoints, credentials, and other large/private files out of Git.

## Milestones and current status

### M0. Local setup and GitHub repository

- Confirm OS, Python/Conda setup, editor, GPU access, and GitHub status.
- Install/check Git and create the project directory.
- Create the GitHub repository and first commit.
- **Status:** Completed.
- **Exit criteria:** Repository exists, local Git works, and initial documentation is committed.

### M1. Python environment

- Create a dedicated Conda environment.
- Install PyTorch, torchvision, and basic utilities.
- Record package versions and whether CUDA is available.
- **Status:** Completed.
- **Environment:** Windows 10, Python 3.10.22, PyTorch 2.5.1+cu121, TorchVision 0.20.1+cu121, NVIDIA GeForce GTX 1050 (4 GB), Conda environment `adv-robust`.
- **Exit criteria:** Imports work and diagnostic CUDA operations run successfully.

### M2. Dataset preparation

- Download and inspect CIFAR-10 and CIFAR-100.
- Verify sample images, labels, tensor shapes, class counts, train/test splits, and data loading.
- **Status:** Completed for the experimental pipeline.
- **Exit criteria:** Both datasets load and batches pass validation.

### M3. Clean baselines

- Implement and train a CIFAR-appropriate SimpleCNN baseline.
- Extend the experiments to a CIFAR-adapted ResNet-18.
- Record training configurations, checkpoints, and metrics.
- **Status:** Initial baselines completed.
- **Completed work:** SimpleCNN baselines on CIFAR-10 and CIFAR-100; CIFAR-adapted ResNet-18 trained on the full 50,000-image CIFAR-10 training set for 100 epochs.
- **Important limitation:** The reported ResNet-18 checkpoint was selected using test accuracy during training. Its 89.26% test accuracy is subject to test-set selection bias. Retrain with validation-only checkpoint selection before treating it as a rigorous final baseline.
- **Exit criteria:** Training and evaluation run end to end and results are recorded. A methodologically clean final baseline remains to be established.

### M4. Attack evaluation

- Implement FGSM and PGD with explicit epsilon, step size, step count, and random-start settings.
- Check pixel bounds and the L-infinity perturbation constraint.
- Evaluate clean, FGSM, and PGD accuracy using a fixed test protocol.
- Compare custom attack implementations with an established implementation for selected settings.
- Report clean accuracy, adversarial accuracy, and ASR.
- **Status:** Core implementation and evaluation completed for the tested experiments.
- **Completed work:** FGSM and PGD evaluations on CIFAR-10/CIFAR-100 SimpleCNN experiments and CIFAR-10 ResNet-18; custom ResNet-18 FGSM/PGD outputs compared with `torchattacks` for the tested settings; PGD-50 with five random restarts run for selected ResNet-18 models.
- **Remaining:** Independently audit ASR calculations and denominators; repeat key evaluations across seeds and, where needed, stronger/multiple-restart settings.
- **Exit criteria:** Attack implementations pass the recorded checks and evaluation metrics are saved. Broader validation remains ongoing.

### M5. Adversarial training

- Implement a basic adversarial-training loop.
- Compare clean and adversarially trained models using the same evaluation protocol.
- Track clean and robust accuracy, training settings, and relevant trade-offs.
- **Status:** Initial FGSM adversarial-training experiments completed.
- **Completed work:** FGSM adversarial training on CIFAR-10/CIFAR-100 SimpleCNN and CIFAR-10 ResNet-18; clean, FGSM, and PGD comparisons recorded.
- **Scope note:** The current work does not establish a completed PGD-adversarial-training comparison.
- **Remaining:** Multi-seed validation and a clean ResNet-18 baseline/model-selection protocol.
- **Exit criteria:** Initial comparisons are documented; reproducibility and methodology improvements remain.

### M6. CIFAR-100 extension

- Adapt the model output layer and repeat selected experiments.
- Discuss the added difficulty from 100 classes.
- **Status:** Exploratory CIFAR-100 experiments completed.
- **Completed work:** SimpleCNN clean baseline, FGSM/PGD evaluation, and FGSM adversarial training.
- **Limitations:** Five training epochs and one random seed; these results are not a mature or multi-seed benchmark.
- **Remaining:** Longer training and multi-seed validation if CIFAR-100 remains part of the final research scope.
- **Exit criteria:** The exploratory pipeline runs and results are logged; reproducibility criteria are not yet met.

### M7. HYDRA code study and reproduction

- Read the official HYDRA paper, README, dependencies, data pipeline, models, trainer, and scripts.
- Inspect its expected environment before installing legacy dependencies.
- Run limited smoke tests in an isolated reference-code environment.
- **Status:** Paper/code study and smoke tests completed; full reproduction not completed.
- **Completed work:** Inspected the reference implementation and ran limited CIFAR-10 smoke tests using a 10% data subset and short runs. Tested pipeline stages and verified a 50% sparse checkpoint in the smoke-test setup.
- **Important distinction:** These checks confirm that selected code paths execute; they do not reproduce the paper's experimental results or establish a fair comparison with our models.
- **Remaining:** A controlled full reproduction only if it is necessary for the final research question and feasible with available resources.
- **Exit criteria:** For a future reproduction, document the original settings, deviations, datasets, training duration, and comparable metrics. This exit criterion is not currently met.

### M8. Pruning comparison, efficiency measurement, and research question

- Evaluate global magnitude pruning at 30%, 50%, and 70% sparsity.
- Compare clean accuracy, adversarial accuracy, and ASR before and after fine-tuning.
- Measure actual model storage and inference performance, while distinguishing unstructured sparsity from practical acceleration.
- Use the results to formulate a testable research question.
- **Status:** Initial pruning, fine-tuning, and efficiency experiments completed; research contribution not yet finalized.
- **Completed work:** SimpleCNN and ResNet-18 magnitude-pruning experiments; fine-tuning comparisons; PGD-50 five-restart evaluation of selected ResNet-18 models; preliminary batch-64 inference benchmark on a GTX 1050.
- **Current finding:** Pruning did not produce a consistent robustness improvement. The focused inference benchmark did not demonstrate meaningful latency or dense checkpoint-size reduction. No sparse inference kernel or compressed sparse storage was evaluated.
- **Limitations:** Most comparisons use one seed and one checkpoint per configuration. The efficiency benchmark used one GPU, one batch size, and ten measurement rounds.
- **Remaining:** Multi-seed validation, research-gap definition, and a proposal/report grounded in the results. Quantization is optional and should be considered only if it helps answer the final research question.
- **Exit criteria:** A reproducible comparison and a short evidence-based proposal are still outstanding.

## Initial evaluation protocol

- Start with CIFAR-10; use CIFAR-100 as an exploratory extension unless the final research question requires more.
- Report clean accuracy and adversarial accuracy separately.
- Report ASR with an explicit definition and denominator; in the current ResNet-18 PGD-50 experiment, cumulative ASR counts initially correctly classified images successfully attacked by at least one restart.
- Record attack norm, epsilon, step size, steps, random-start setting, number of restarts, and evaluation sample count.
- Keep training, validation, and test data roles explicit. Use validation data for checkpoint/model selection and reserve test data for final evaluation.
- Compare models under matched attack settings.
- Report sparsity separately from actual compressed storage, memory use, and inference speed.
- Use a small smoke test before long runs and label smoke tests clearly.

## Current priority order

1. Repeat key training, pruning, and fine-tuning experiments across multiple independent seeds.
2. Retrain the ResNet-18 baseline and select checkpoints using validation data only.
3. Independently audit ASR calculations and evaluation denominators.
4. Consolidate the experimental tables, figures, settings, and limitations.
5. Define a specific research gap and testable contribution with the supervisor.
6. Decide whether a full HYDRA reproduction or quantization experiment is necessary for that research question.
7. Prepare the research report.

## Key references

- HYDRA: https://arxiv.org/abs/2002.10509 · https://github.com/inspire-group/hydra
- Robustness-Aware-Pruning-ADMM: https://github.com/yeshaokai/Robustness-Aware-Pruning-ADMM
- CAGE: https://github.com/XinweiZhang1998/CAGE
- CIFAR-10/100: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch: https://pytorch.org/
