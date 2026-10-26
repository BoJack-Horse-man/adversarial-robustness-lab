
"""Compare baseline and 50%-pruned models using multi-restart PGD-50."""

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
RESULT_PATH = Path("experiments/pruned_pgd50_5restarts_comparison.csv")

CHECKPOINTS = {
    "baseline": Path("checkpoints/simple_cnn.pth"),
    "pruned_50_percent": Path("checkpoints/simple_cnn_pruned_50.pth"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pgd_attack(model, images, labels):
    original = images.detach()

    adversarial = (
        original
        + torch.empty_like(original).uniform_(-EPSILON, EPSILON)
    ).clamp(0, 1).detach()

    for _ in range(PGD_STEPS):
        adversarial.requires_grad_(True)
        loss = nn.functional.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]

        adversarial = adversarial.detach() + ALPHA * gradient.sign()

        delta = (adversarial - original).clamp(-EPSILON, EPSILON)
        adversarial = (original + delta).clamp(0, 1).detach()

    return adversarial


def evaluate_model(model, loader, model_seed):
    # Use a reproducible, model-specific random stream.
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    clean_correct = 0
    robust_correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)
        clean_correct += (clean_predictions == labels).sum().item()

        # Robust means correct under every random restart.
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

    return 100 * clean_correct / total, 100 * robust_correct / total


def main():
    for path in CHECKPOINTS.values():
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

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

    print(f"Device: {DEVICE}")
    print(f"Test images: {len(dataset)}")
    print(f"PGD steps: {PGD_STEPS}; restarts: {RESTARTS}")
    print(f"Epsilon: {EPSILON:.8f}; step size: {ALPHA:.8f}")

    results = []

    for index, (name, checkpoint_path) in enumerate(CHECKPOINTS.items()):
        model = SimpleCNN().to(DEVICE)
        state = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()

        clean_accuracy, robust_accuracy = evaluate_model(
            model, loader, SEED + index
        )

        result = {
            "model": name,
            "attack": "PGD",
            "epsilon": EPSILON,
            "alpha": ALPHA,
            "steps": PGD_STEPS,
            "restarts": RESTARTS,
            "test_images": len(dataset),
            "clean_accuracy_percent": f"{clean_accuracy:.2f}",
            "robust_accuracy_percent": f"{robust_accuracy:.2f}",
        }
        results.append(result)

        print(f"\n{name}")
        print(f"  Clean accuracy:  {clean_accuracy:.2f}%")
        print(f"  Robust accuracy: {robust_accuracy:.2f}%")

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    baseline_robust = float(results[0]["robust_accuracy_percent"])
    pruned_robust = float(results[1]["robust_accuracy_percent"])

    print(
        "\nRobust accuracy change after pruning: "
        f"{pruned_robust - baseline_robust:+.2f} percentage points"
    )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
