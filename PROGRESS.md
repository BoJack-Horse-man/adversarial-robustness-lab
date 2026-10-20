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

## FGSM Attack Evaluation - Completed

### Setup
- Model: SimpleCNN, trained for 5 epochs
- Dataset: CIFAR-10 test set (10,000 images)
- Evaluation: untargeted FGSM
- Pixel range: [0, 1]
- Checkpoint: checkpoints/simple_cnn.pth

### Results
| Condition | Epsilon | Test accuracy |
|---|---:|---:|
| Clean | 0 | 55.53% |
| FGSM | 2/255 | 34.57% |
| FGSM | 4/255 | 20.54% |
| FGSM | 8/255 | 7.61% |

### Interpretation
Test accuracy decreased as the FGSM perturbation strength increased. At epsilon = 8/255, accuracy fell to 7.61%, indicating that this baseline model is vulnerable to this attack under the tested settings.

These are preliminary results from one trained model and one random seed. They do not establish general robustness conclusions.

### Artifact
- Evaluation script: eval_fgsm.py
- Results: experiments/fgsm_results.csv

### Next task
Implement Projected Gradient Descent (PGD), evaluate the same checkpoint, and compare the results under clearly specified attack settings.

## PGD Attack Evaluation

**Status:** Completed

### Experimental setup
- Dataset: CIFAR-10 test set (10,000 images)
- Model: SimpleCNN trained for 5 epochs
- Device: NVIDIA GPU (`cuda`)
- Attack: Untargeted Projected Gradient Descent (PGD)
- Maximum perturbation: ε = 8/255
- Step size: α = 2/255
- Iterations: 10
- Random start: Enabled
- Pixel range: [0, 1]

### Results

| Evaluation | Test accuracy |
|---|---:|
| Clean images | 55.53% |
| FGSM, ε = 8/255 | 7.61% |
| PGD, ε = 8/255 | 3.49% |

PGD reduced the model's accuracy by 52.04 percentage points compared with clean evaluation. Under these experimental settings, PGD reduced accuracy further than FGSM at the same maximum perturbation.

### Interpretation
The model is vulnerable to gradient-based adversarial attacks. The iterative PGD attack was more effective than FGSM in this experiment. However, the model's clean accuracy is only 55.53%, so further experiments are needed before drawing broader conclusions about robustness.

### Output
- Script: `eval_pgd.py`
- Results: `experiments/pgd_results.csv`

### Next step
Compare the attacks systematically, then implement adversarial training and evaluate whether it improves robustness against FGSM and PGD.


## Completed experiments

- [x] Set up the PyTorch and CUDA environment.
- [x] Download and inspect CIFAR-10.
- [x] Train the baseline SimpleCNN.
- [x] Evaluate the baseline using FGSM.
- [x] Evaluate the baseline using PGD.
- [x] Train a separate model using FGSM adversarial training.
- [x] Compare baseline and adversarially trained models.
- [ ] Prepare and evaluate CIFAR-100.
- [ ] Study and reproduce the relevant parts of HYDRA.
- [ ] Investigate pruning and robustness trade-offs.

## Experiment results

Dataset: CIFAR-10 test set (10,000 images)

Baseline model:
- Clean accuracy: 55.53%
- FGSM, epsilon 2/255: 34.57%
- FGSM, epsilon 4/255: 20.54%
- FGSM, epsilon 8/255: 7.61%
- PGD, epsilon 8/255, 10 steps: 3.46%

FGSM adversarially trained model:
- Training: 5 epochs, batch size 64
- Training epsilon: 8/255
- Clean accuracy: 45.88%
- FGSM, epsilon 2/255: 41.39%
- FGSM, epsilon 4/255: 37.12%
- FGSM, epsilon 8/255: 28.87%
- PGD, epsilon 8/255, 10 steps: 27.68%

Preliminary observation:
Adversarial training improved accuracy under the tested FGSM
and PGD attacks but reduced clean test accuracy. More experiments
are needed to assess reproducibility and generalization.

Result files:
- experiments/baseline_result.txt
- experiments/fgsm_results.csv
- experiments/pgd_results.csv
- experiments/adversarial_training_result.txt
- experiments/adversarial_training_history.csv
- experiments/model_robustness_comparison.csv


## Experiment results
No experiments run yet.

Use `experiments/results.csv` to record model, dataset, training method, seed, attack settings, clean accuracy, robust accuracy, sparsity, runtime, and notes.
