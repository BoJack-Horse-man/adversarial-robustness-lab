# Research Progress Log

**Project:** Adversarial Robustness of Efficient Neural Networks  
**Status:** Planning  
**Last updated:** 2026-10-19

## Current position
- [x] Research goal and broad milestones drafted.
- [x] HYDRA selected as the first reference repository.
- [ ] Confirm local computer and development setup.
- [ ] Create and connect the personal GitHub repository.
- [ ] Set up the Python environment.
- [ ] Download and verify CIFAR-10/CIFAR-100.
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

## Experiment results
No experiments run yet.

Use `experiments/results.csv` to record model, dataset, training method, seed, attack settings, clean accuracy, robust accuracy, sparsity, runtime, and notes.
