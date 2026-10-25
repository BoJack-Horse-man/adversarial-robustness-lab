"""Evaluate CIFAR-10 robustness using multi-restart PGD."""

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from eval_pgd import SimpleCNN

SEED = 42
BATCH_SIZE = 64
EPSILON = 8 / 255
ALPHA = 2 / 255
PGD_STEPS = 50
RESTARTS = 5

DATA_DIR = Path("data")
CHECKPOINT_PATH = Path("checkpoints/simple_cnn.pth")
RESULT_PATH = Path("experiments/pgd50_5restarts_results.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pgd_attack(model, images, labels):
    """Run PGD with a random start and return adversarial images."""
    original = images.detach()
    adversarial = (
        original
        + torch.empty_like(original).uniform_(-EPSILON, EPSILON)
    ).clamp(0, 1).detach()

    for _ in range(PGD_STEPS):
        adversarial.requires_grad_(True)
        loss = nn.functional.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]

        adversarial = (
            adversarial.detach() + ALPHA * gradient.sign()
        )

        delta = (adversarial - original).clamp(-EPSILON, EPSILON)
        adversarial = (original + delta).clamp(0, 1).detach()

    return adversarial


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = SimpleCNN().to(DEVICE)
    state = torch.load(
        CHECKPOINT_PATH, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    clean_correct = 0
    robust_correct = 0
    total = 0

    print(f"Device: {DEVICE}")
    print(f"Test images: {len(dataset)}")
    print(f"PGD steps: {PGD_STEPS}; restarts: {RESTARTS}")
    print(f"Epsilon: {EPSILON:.8f}; step size: {ALPHA:.8f}")

    for images, labels in tqdm(loader, desc="Multi-restart PGD"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)
        clean_correct += (clean_predictions == labels).sum().item()

        # A sample is robust only if every restart still predicts correctly.
        robust_for_batch = torch.ones(
            labels.size(0), dtype=torch.bool, device=DEVICE
        )

        for _ in range(RESTARTS):
            adversarial = pgd_attack(model, images, labels)
            with torch.no_grad():
                predictions = model(adversarial).argmax(dim=1)
            robust_for_batch &= predictions.eq(labels)

        robust_correct += robust_for_batch.sum().item()
        total += labels.size(0)

    clean_accuracy = 100 * clean_correct / total
    robust_accuracy = 100 * robust_correct / total

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "attack", "epsilon", "alpha", "steps", "restarts",
            "clean_accuracy_percent", "robust_accuracy_percent",
        ])
        writer.writerow([
            "PGD", EPSILON, ALPHA, PGD_STEPS, RESTARTS,
            f"{clean_accuracy:.2f}", f"{robust_accuracy:.2f}",
        ])

    print("\n--- Multi-restart PGD Results ---")
    print(f"Clean accuracy: {clean_accuracy:.2f}%")
    print(f"Robust accuracy: {robust_accuracy:.2f}%")
    print(f"Saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
