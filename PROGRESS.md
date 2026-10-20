# Research Progress Log

**Project:** Adversarial Robustness of Efficient Neural Networks  
**Status:** Planning  
**Last updated:** 2026-10-20

## Current position
- [x] Research goal and broad milestones drafted.
- [x] HYDRA selected as the first reference repository.
- [x] Confirm local computer and development setup.
- [x] Create and connect the personal GitHub repository.
- [x] Set up the Python environment.
- [X] Download and verify CIFAR-10/CIFAR-100.
- [ ] Train a clean baseline.
- [ ] Implement and validate FGSM/PGD.
- [ ] Run adversarial-training experiments.
- [ ] Study/reproduce a HYDRA experiment.
- [ ] Compare pruning and robustness results.

## Session log

### 2026-10-19: Planning
**Completed**
- Defined a staged workflow from environment setup to pruning research.
- Chose CIFAR-10 as the first working dataset and CIFAR-100 as the follow-up.
- Selected HYDRA as the first codebase to study.

**Decisions**
- Work one step at a time; do not proceed until the current step is verified.
- Keep our code in a separate repository from the reference implementations.
- Start with a modern baseline, then inspect HYDRA's older dependencies in isolation.
- Track experiment settings and results instead of relying on memory.

**Open questions**
- Operating system and editor?
- Conda/Python status?
- Local NVIDIA GPU or cloud GPU availability?
- GitHub account/repository status?
- Git installation and authentication status?

**Next task**
- Confirm setup details, then create the local project folder and GitHub repository.


## Environment Setup - Completed 2026-10-20

- Created the Conda environment `adv-robust`.
- Python version: 3.10.22.
- PyTorch version: 2.5.1+cu121.
- TorchVision version: 0.20.1+cu121.
- NumPy version: 2.2.6.
- Matplotlib version: 3.10.9.
- tqdm version: 4.70.1.
- GPU: NVIDIA GeForce GTX 1050 with 4 GB VRAM.
- Verified that `torch.cuda.is_available()` returns `True`.
- Successfully performed a matrix multiplication on `cuda:0`.
- GitHub repository initialized and connected to the remote repository.

### Next tasks
1. Verify CIFAR-10 dataset downloading and loading.
2. Build a simple baseline image classifier.
3. Train and record clean test accuracy.
4. Implement and evaluate FGSM and PGD attacks.
5. Explore adversarial training and efficient-model defenses.

## First CIFAR-10 Baseline - Completed

### Environment
- Python: 3.10.22
- PyTorch: 2.5.1+cu121
- GPU: NVIDIA GeForce GTX 1050, 4 GB VRAM

### Dataset
- CIFAR-10
- Training images: 50,000
- Test images: 10,000
- Transform: ToTensor, scaling pixel values to [0, 1]

### Model
- Architecture: SimpleCNN
- Trainable parameters: 94,538
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Epochs: 5
- Random seed: 42

### Results
- Epoch 1 training accuracy: 31.50%
- Epoch 5 training accuracy: 53.85%
- Epoch 5 average training loss: 1.2739
- Clean test accuracy: 55.53%

### Artifacts
- Training script: train_baseline.py
- Dataset check: check_cifar10.py
- Dataset visualization: inspect_cifar10.py
- Results record: experiments/baseline_result.txt
- Model weights: checkpoints/simple_cnn.pth (local, not tracked by Git)

### Next tasks
1. Record the baseline in the experiment results table.
2. Implement FGSM and evaluate clean versus adversarial accuracy.
3. Implement PGD and compare attack strength.
4. Investigate adversarial training and efficient-model defenses.

Note: This is a preliminary five-epoch baseline. It is not yet a final benchmark or a multi-seed result.

## Experiment results
No experiments run yet.

Use `experiments/results.csv` to record model, dataset, training method, seed, attack settings, clean accuracy, robust accuracy, sparsity, runtime, and notes.
