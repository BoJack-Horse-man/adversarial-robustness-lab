
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# 1. Set random seeds for more reproducible experiments.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 2. Load CIFAR-10.
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform,
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform,
)

# A smaller batch is suitable for a GPU with 4 GB of memory.
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=0
)


# 3. Define a small CNN.
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

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Model parameters:", sum(p.numel() for p in model.parameters()))


# 4. Train for five epochs.
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


# 5. Evaluate on the separate test set.
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

print(f"\nClean test accuracy: {test_accuracy:.2f}%")


# 6. Save the model and a small results record.
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

torch.save(model.state_dict(), "checkpoints/simple_cnn.pth")

with open("experiments/baseline_result.txt", "w", encoding="utf-8") as file:
    file.write(f"Seed: {SEED}\n")
    file.write(f"Device: {device}\n")
    file.write(f"Epochs: {EPOCHS}\n")
    file.write(f"Batch size: 64\n")
    file.write(f"Optimizer: Adam\n")
    file.write(f"Learning rate: 0.001\n")
    file.write(f"Clean test accuracy: {test_accuracy:.2f}%\n")

print("Saved model to checkpoints/simple_cnn.pth")
print("Saved results to experiments/baseline_result.txt")
