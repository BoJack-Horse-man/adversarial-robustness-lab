
import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# Experiment settings
BATCH_SIZE = 64
EPSILONS = [0, 2 / 255, 4 / 255, 8 / 255]

DATA_DIR = "./data_cifar100"
CHECKPOINT_PATH = "./checkpoints/simple_cnn_cifar100_adv.pth"
OUTPUT_PATH = "./experiments/cifar100_fgsm_adv_training_results.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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


# Load model and test data
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


def evaluate_fgsm(epsilon):
    model.eval()
    correct = 0
    total = 0

    for images, labels in tqdm(test_loader, desc=f"FGSM eps={epsilon:.5f}"):
        images = images.to(device)
        labels = labels.to(device)

        # Compute the gradient with respect to input images
        attack_images = images.detach().clone().requires_grad_(True)
        outputs = model(attack_images)
        loss = criterion(outputs, labels)

        gradients = torch.autograd.grad(loss, attack_images)[0]

        # FGSM perturbation, clipped to valid image range
        adversarial_images = (
            images + epsilon * gradients.sign()
        ).clamp(0.0, 1.0).detach()

        # Evaluate adversarial examples
        with torch.no_grad():
            predictions = model(adversarial_images).argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


results = []

for epsilon in EPSILONS:
    accuracy = evaluate_fgsm(epsilon)
    results.append({
        "epsilon": epsilon,
        "epsilon_fraction": f"{epsilon * 255:.0f}/255",
        "accuracy_percent": accuracy,
    })
    print(
        f"Epsilon {epsilon:.6f} "
        f"({epsilon * 255:.0f}/255): accuracy = {accuracy:.2f}%"
    )

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["epsilon", "epsilon_fraction", "accuracy_percent"],
    )
    writer.writeheader()
    writer.writerows(results)

print("Results saved to:", OUTPUT_PATH)
