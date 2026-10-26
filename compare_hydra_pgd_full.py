import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Use the separate HYDRA reference repository without modifying it.
sys.path.insert(0, str(Path(r"..\hydra-reference").resolve()))

import models
from utils.model import get_layers, prepare_model
from utils.adv import pgd_whitebox


SEED = 2026
EPSILON = 8 / 255
STEP_SIZE = 2 / 255
NUM_STEPS = 10
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = Path(
    r"..\hydra-pilot-results\cifar10-subnet-pretrain\pretrain"
    r"\latest_exp\checkpoint\model_best.pth.tar"
)
DATA_DIR = Path(r"..\hydra-pilot-data\CIFAR10")


def hydra_pgd_from_start(model, x, y, x_start):
    """Reproduce HYDRA's update steps from a supplied starting image."""
    x_original = x.detach()
    x_adv = x_start.detach().requires_grad_(True)

    for _ in range(NUM_STEPS):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = (x_adv.detach() + STEP_SIZE * grad.sign()).detach()

        delta = torch.clamp(
            x_adv - x_original, min=-EPSILON, max=EPSILON
        )
        x_adv = (x_original + delta).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
        x_adv.requires_grad_(True)

    return x_adv.detach()


def corrected_pgd(model, x, y, x_start):
    """PGD from the same start, with valid initialization and projection."""
    x_original = x.detach()
    x_adv = torch.clamp(x_start.detach(), 0.0, 1.0)

    # Project the supplied start onto the valid epsilon ball.
    delta = torch.clamp(
        x_adv - x_original, min=-EPSILON, max=EPSILON
    )
    x_adv = torch.clamp(x_original + delta, 0.0, 1.0).detach()

    for _ in range(NUM_STEPS):
        x_adv = x_adv.detach().requires_grad_(True)

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + STEP_SIZE * grad.sign()
        delta = torch.clamp(
            x_adv - x_original, min=-EPSILON, max=EPSILON
        )
        x_adv = torch.clamp(x_original + delta, 0.0, 1.0).detach()

    return x_adv

def accuracy(model, x, y):
    with torch.no_grad():
        predictions = model(x).argmax(dim=1)
    return (predictions == y).float().mean().item() * 100


def audit(name, x_adv, x, y, model):
    delta = (x_adv - x).abs()
    print(f"\n{name}")
    print(f"  Adversarial accuracy: {accuracy(model, x_adv, y):.2f}%")
    print(f"  Maximum perturbation: {delta.max().item():.8f}")
    print(f"  Epsilon limit:       {EPSILON:.8f}")
    print(f"  Within epsilon:       {delta.max().item() <= EPSILON + 1e-6}")
    print(
        "  Pixels in [0,1]:     "
        f"{x_adv.min().item() >= 0 and x_adv.max().item() <= 1}"
    )
    print(f"  Finite values:        {torch.isfinite(x_adv).all().item()}")


def main():
    import csv

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT.resolve()}")
    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(f"CIFAR-10 data not found: {DATA_DIR.resolve()}")

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT.resolve())
    print("Dataset:", DATA_DIR.resolve())
    print(f"Attack: epsilon={EPSILON:.8f}, step_size={STEP_SIZE:.8f}, steps={NUM_STEPS}")

    args = SimpleNamespace(
        layer_type="subnet",
        exp_mode="pretrain",
        k=1.0,
        freeze_bn=False,
        scores_init_type="kaiming_normal",
    )
    conv, linear = get_layers("subnet")
    model = models.cifar_model(
        conv, linear, "kaiming_normal", num_classes=10
    ).to(DEVICE)
    prepare_model(model, args)

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    total = 0
    clean_correct = 0
    original_correct = 0
    corrected_correct = 0
    different_batches = 0
    prediction_disagreements = 0
    correctness_disagreements = 0
    max_perturbation_original = 0.0
    max_perturbation_corrected = 0.0
    all_pixels_valid = True
    all_values_finite = True

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    for batch_index, (x, y) in enumerate(loader, start=1):
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.no_grad():
            clean_correct += (model(x).argmax(dim=1) == y).sum().item()

        # Match HYDRA's CPU-generated random noise, shared by both attacks.
        noise = torch.empty(
            x.shape, dtype=x.dtype, device="cpu"
        ).uniform_(-EPSILON, EPSILON).to(DEVICE)
        raw_start = (x.detach() + noise).detach()

        original_adv = hydra_pgd_from_start(model, x, y, raw_start)
        corrected_adv = corrected_pgd(model, x, y, raw_start)

        original_delta = (original_adv - x).abs()
        corrected_delta = (corrected_adv - x).abs()

        with torch.no_grad():
            original_preds = model(original_adv).argmax(dim=1)
            corrected_preds = model(corrected_adv).argmax(dim=1)
            original_is_correct = original_preds == y
            corrected_is_correct = corrected_preds == y

        original_correct += original_is_correct.sum().item()
        corrected_correct += corrected_is_correct.sum().item()
        prediction_disagreements += (
            original_preds != corrected_preds
        ).sum().item()
        correctness_disagreements += (
            original_is_correct != corrected_is_correct
        ).sum().item()

        max_perturbation_original = max(
            max_perturbation_original, original_delta.max().item()
        )
        max_perturbation_corrected = max(
            max_perturbation_corrected, corrected_delta.max().item()
        )

        all_pixels_valid = all_pixels_valid and (
            original_adv.min().item() >= 0
            and original_adv.max().item() <= 1
            and corrected_adv.min().item() >= 0
            and corrected_adv.max().item() <= 1
        )
        all_values_finite = all_values_finite and (
            torch.isfinite(original_adv).all().item()
            and torch.isfinite(corrected_adv).all().item()
        )

        if not torch.equal(original_adv, corrected_adv):
            different_batches += 1

        total += y.size(0)

        if batch_index % 100 == 0 or total == len(dataset):
            print(f"Processed {total}/{len(dataset)} images")

    results = [
        ("test_images", total),
        ("clean_accuracy_percent", 100 * clean_correct / total),
        ("original_hydra_accuracy_percent", 100 * original_correct / total),
        ("corrected_pgd_accuracy_percent", 100 * corrected_correct / total),
        ("original_max_perturbation", max_perturbation_original),
        ("corrected_max_perturbation", max_perturbation_corrected),
        ("epsilon", EPSILON),
        ("all_final_pixels_in_0_1", all_pixels_valid),
        ("all_final_values_finite", all_values_finite),
        ("batches_with_different_adversarial_images", different_batches),
        ("prediction_disagreements", prediction_disagreements),
        ("prediction_disagreement_percent", 100 * prediction_disagreements / total),
        ("correctness_disagreements", correctness_disagreements),
        ("correctness_disagreement_percent", 100 * correctness_disagreements / total),
        ("batch_size", BATCH_SIZE),
        ("random_seed", SEED),
        ("pgd_steps", NUM_STEPS),
    ]

    output_path = Path("hydra_pgd_full_comparison_audit.csv")
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(results)

    print("\nFull CIFAR-10 test results")
    for key, value in results:
        print(f"{key}: {value}")
    print(f"\nSaved results to: {output_path.resolve()}")
if __name__ == "__main__":
    main()
