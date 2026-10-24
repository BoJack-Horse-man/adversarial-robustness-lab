
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# 1. Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 2. Load CIFAR-100
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR100(
    root="./data_cifar100",
    train=True,
    download=False,
    transform=transform,
)

test_dataset = datasets.CIFAR100(
    root="./data_cifar100",
    train=False,
    download=False,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=0
)


# 3. Define the CNN
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

        # CIFAR-100 has 100 classes, not 10.
        self.classifier = nn.Linear(128, 100)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))
print("Number of classes:", len(train_dataset.classes))
print("Model parameters:", sum(p.numel() for p in model.parameters()))


# 4. Train for five epochs
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        progress.set_postfix(
            loss=f"{loss.item():.3f}",
            accuracy=f"{100 * correct / total:.1f}%"
        )

    average_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}: "
        f"average loss={average_loss:.4f}, "
        f"training accuracy={train_accuracy:.2f}%"
    )


# 5. Evaluate on the test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

test_accuracy = 100 * correct / total

print(f"\nCIFAR-100 clean test accuracy: {test_accuracy:.2f}%")


# 6. Save to CIFAR-100-specific paths
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

checkpoint_path = "checkpoints/simple_cnn_cifar100.pth"
results_path = "experiments/cifar100_baseline_result.txt"

torch.save(model.state_dict(), checkpoint_path)

with open(results_path, "w", encoding="utf-8") as file:
    file.write(f"Dataset: CIFAR-100\n")
    file.write(f"Seed: {SEED}\n")
    file.write(f"Device: {device}\n")
    file.write(f"Epochs: {EPOCHS}\n")
    file.write("Training batch size: 64\n")
    file.write("Test batch size: 128\n")
    file.write("Optimizer: Adam\n")
    file.write("Learning rate: 0.001\n")
    file.write(f"Clean test accuracy: {test_accuracy:.2f}%\n")

print("Saved model to", checkpoint_path)
print("Saved results to", results_path)
