
"""Evaluate a trained CIFAR-100 CNN against a PGD attack."""

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# 1. Configuration
SEED = 42
BATCH_SIZE = 64

EPSILON = 8 / 255
ALPHA = 2 / 255
PGD_STEPS = 10

DATA_DIR = Path("data_cifar100")
CHECKPOINT_PATH = Path("checkpoints/simple_cnn_cifar100.pth")
RESULT_PATH = Path("experiments/cifar100_pgd_results.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# 2. Same architecture used for CIFAR-100 training
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, 100)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


# 3. PGD attack
def pgd_attack(model, images, labels, epsilon, alpha, steps):
    """Create untargeted PGD adversarial examples."""

    original_images = images.detach()

    # Random start within the epsilon range.
    noise = torch.empty_like(original_images).uniform_(
        -epsilon, epsilon
    )
    adversarial_images = (
        original_images + noise
    ).clamp(0, 1).detach()

    for _ in range(steps):
        adversarial_images.requires_grad_(True)

        outputs = model(adversarial_images)
        loss = nn.functional.cross_entropy(outputs, labels)

        gradient = torch.autograd.grad(
            loss, adversarial_images
        )[0]

        # Gradient ascent to increase classification loss.
        adversarial_images = (
            adversarial_images.detach()
            + alpha * gradient.sign()
        )

        # Project perturbation into the epsilon ball.
        perturbation = torch.clamp(
            adversarial_images - original_images,
            min=-epsilon,
            max=epsilon,
        )

        adversarial_images = (
            original_images + perturbation
        ).clamp(0, 1).detach()

    return adversarial_images


# 4. Load test data and checkpoint
def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    test_dataset = datasets.CIFAR100(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = SimpleCNN().to(DEVICE)

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    print("Device:", DEVICE)
    print("Test images:", len(test_dataset))
    print(f"Epsilon: {EPSILON:.6f} ({EPSILON * 255:.0f}/255)")
    print(f"Step size: {ALPHA:.6f} ({ALPHA * 255:.0f}/255)")
    print("PGD steps:", PGD_STEPS)
    print("Random start: enabled")

    # 5. Evaluate clean and adversarial accuracy
    clean_correct = 0
    adversarial_correct = 0
    total = 0

    for images, labels in tqdm(
        test_loader, desc="CIFAR-100 PGD evaluation"
    ):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            clean_outputs = model(images)
            clean_predictions = clean_outputs.argmax(dim=1)
            clean_correct += (
                clean_predictions == labels
            ).sum().item()

        adversarial_images = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=EPSILON,
            alpha=ALPHA,
            steps=PGD_STEPS,
        )

        with torch.no_grad():
            adversarial_outputs = model(adversarial_images)
            adversarial_predictions = adversarial_outputs.argmax(
                dim=1
            )
            adversarial_correct += (
                adversarial_predictions == labels
            ).sum().item()

        total += labels.size(0)

    clean_accuracy = 100 * clean_correct / total
    pgd_accuracy = 100 * adversarial_correct / total
    accuracy_drop = clean_accuracy - pgd_accuracy

    print("\n--- CIFAR-100 PGD Results ---")
    print(f"Clean test accuracy: {clean_accuracy:.2f}%")
    print(f"PGD adversarial accuracy: {pgd_accuracy:.2f}%")
    print(f"Accuracy decrease: {accuracy_drop:.2f} percentage points")

    # 6. Save to a CIFAR-100-specific results file
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "dataset",
            "attack",
            "epsilon",
            "alpha",
            "steps",
            "random_start",
            "clean_accuracy_percent",
            "adversarial_accuracy_percent",
            "accuracy_drop_percentage_points",
        ])
        writer.writerow([
            "CIFAR-100",
            "PGD",
            EPSILON,
            ALPHA,
            PGD_STEPS,
            True,
            f"{clean_accuracy:.2f}",
            f"{pgd_accuracy:.2f}",
            f"{accuracy_drop:.2f}",
        ])

    print("Results saved to:", RESULT_PATH)


if __name__ == "__main__":
    main()
