
import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# Experiment settings
BATCH_SIZE = 64
EPSILON = 8 / 255
ALPHA = 2 / 255
STEPS = 10

DATA_DIR = "./data_cifar100"
CHECKPOINT_PATH = "./checkpoints/simple_cnn_cifar100_adv.pth"
OUTPUT_PATH = "./experiments/cifar100_pgd_adv_training_results.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(f"PGD epsilon: {EPSILON:.6f}")
print(f"PGD step size: {ALPHA:.6f}")
print(f"PGD steps: {STEPS}")
print("Random start: enabled")


# Model architecture must match the trained checkpoint
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
        x = torch.flatten(x, 1)
        return self.classifier(x)


model = SimpleCNN().to(device)
model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
)
model.eval()

test_dataset = datasets.CIFAR100(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=transforms.ToTensor(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

criterion = nn.CrossEntropyLoss()


def evaluate_clean():
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Clean evaluation"):
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def generate_pgd(images, labels):
    original_images = images.detach()

    # Random initialization inside the allowed perturbation range
    adversarial_images = (
        original_images
        + torch.empty_like(original_images).uniform_(-EPSILON, EPSILON)
    ).clamp(0.0, 1.0).detach()

    for _ in range(STEPS):
        adversarial_images.requires_grad_(True)

        outputs = model(adversarial_images)
        loss = criterion(outputs, labels)

        gradients = torch.autograd.grad(loss, adversarial_images)[0]

        with torch.no_grad():
            adversarial_images = adversarial_images + ALPHA * gradients.sign()

            # Project back into the epsilon-ball around the original images
            perturbation = torch.clamp(
                adversarial_images - original_images,
                min=-EPSILON,
                max=EPSILON,
            )
            adversarial_images = (
                original_images + perturbation
            ).clamp(0.0, 1.0)

        adversarial_images = adversarial_images.detach()

    return adversarial_images


def evaluate_pgd():
    correct = 0
    total = 0

    for images, labels in tqdm(test_loader, desc="PGD-10 evaluation"):
        images = images.to(device)
        labels = labels.to(device)

        adversarial_images = generate_pgd(images, labels)

        with torch.no_grad():
            predictions = model(adversarial_images).argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


clean_accuracy = evaluate_clean()
pgd_accuracy = evaluate_pgd()

print(f"\nClean accuracy: {clean_accuracy:.2f}%")
print(f"PGD-10 accuracy: {pgd_accuracy:.2f}%")
print(f"Accuracy drop: {clean_accuracy - pgd_accuracy:.2f} percentage points")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "attack",
            "epsilon",
            "alpha",
            "steps",
            "random_start",
            "accuracy_percent",
        ],
    )
    writer.writeheader()
    writer.writerows([
        {
            "attack": "Clean",
            "epsilon": 0,
            "alpha": 0,
            "steps": 0,
            "random_start": False,
            "accuracy_percent": clean_accuracy,
        },
        {
            "attack": "PGD-10",
            "epsilon": EPSILON,
            "alpha": ALPHA,
            "steps": STEPS,
            "random_start": True,
            "accuracy_percent": pgd_accuracy,
        },
    ])

print("Results saved to:", OUTPUT_PATH)
