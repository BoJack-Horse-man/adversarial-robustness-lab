# Research Plan: Adversarial Robustness of Efficient Neural Networks

## Goal
Build a reproducible CIFAR-10/CIFAR-100 research pipeline, evaluate standard adversarial attacks, learn adversarial training, and then study robustness-aware pruning using HYDRA as the first reference implementation.

## Working principles
- Work one milestone at a time and verify each step before moving on.
- Record commands, environment details, errors, fixes, and results.
- Keep our implementation separate from third-party repositories.
- Never claim an experiment is reproduced until the settings and results have been checked.
- Keep datasets, checkpoints, credentials, and other large/private files out of Git.

## Milestones

### M0. Local setup and GitHub repository
- Confirm OS, Python/Conda setup, editor, GPU access, and GitHub status.
- Install/check Git and create the project directory.
- Create the GitHub repository and first commit.
- **Exit criteria:** repository exists, local Git works, initial documentation is committed.

### M1. Python environment
- Create a dedicated Conda environment.
- Install PyTorch, torchvision, and basic utilities.
- Record package versions and whether CUDA is available.
- **Exit criteria:** imports work and a diagnostic script runs successfully.

### M2. Dataset preparation
- Download CIFAR-10 and CIFAR-100 using torchvision.
- Inspect sample images, labels, tensor shapes, and class counts.
- Verify train/test splits and data loading.
- **Exit criteria:** both datasets load and a small batch passes validation.

### M3. Clean baseline
- Implement a CIFAR-appropriate ResNet-18 baseline.
- Train first on CIFAR-10, initially with a short smoke test.
- Save checkpoints locally and log metrics.
- **Exit criteria:** training and evaluation run end to end; results are recorded.

### M4. Attack evaluation
- Implement FGSM and PGD with explicit epsilon, step size, and step count.
- Check pixel bounds and the L-infinity perturbation constraint.
- Evaluate clean, FGSM, and PGD accuracy using a fixed test protocol.
- **Exit criteria:** attack tests pass and metrics are saved.

### M5. Adversarial training
- Implement a basic adversarial-training loop.
- Compare clean training, FGSM training, and PGD training.
- Track clean and robust accuracy, training time, and settings.
- **Exit criteria:** comparisons use the same evaluation protocol and are documented.

### M6. CIFAR-100 extension
- Adapt the model output layer and repeat selected experiments.
- Discuss the added difficulty from 100 classes.
- **Exit criteria:** selected CIFAR-100 runs are reproducible and logged.

### M7. HYDRA code study and reproduction
- Read the official HYDRA README, dependencies, data pipeline, models, trainer, and scripts.
- Identify its expected environment before installing legacy dependencies.
- Reproduce a small, documented experiment in an isolated environment if feasible.
- **Exit criteria:** a reproduction report distinguishes confirmed results from deviations.

### M8. Pruning comparison and research question
- Compare a baseline pruning method with HYDRA-style robustness-aware pruning where feasible.
- Measure clean/robust accuracy, sparsity, model size, runtime, and compute cost.
- Identify limitations and formulate a testable research question.
- **Exit criteria:** evidence-based comparison and a short proposal for next experiments.

## Initial evaluation protocol
- Start with CIFAR-10; use CIFAR-100 after the pipeline works.
- Report clean accuracy and robust accuracy separately.
- Record attack norm, epsilon, step size, steps, random-start setting, and evaluation sample count.
- Keep train/test separation and model-selection rules explicit.
- Use a small smoke test before long runs.

## Key references
- HYDRA: https://github.com/inspire-group/hydra
- Robustness-Aware-Pruning-ADMM: https://github.com/yeshaokai/Robustness-Aware-Pruning-ADMM
- CAGE: https://github.com/XinweiZhang1998/CAGE
- CIFAR-10/100: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch: https://pytorch.org/
