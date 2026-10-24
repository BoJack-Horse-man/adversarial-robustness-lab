
import os
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# 1. Settings
SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
EPSILON = 8 / 255

DATA_DIR = "./data_cifar100"
CHECKPOINT_DIR = "./checkpoints"
EXPERIMENT_DIR = "./experiments"

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR, "simple_cnn_cifar100_adv.pth"
)
RESULT_PATH = os.path.join(
    EXPERIMENT_DIR, "cifar100_adversarial_training_result.txt"
)
CSV_PATH = os.path.join(
    EXPERIMENT_DIR, "cifar100_adversarial_training_history.csv"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


# 2. Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)
print(f"Adversarial training epsilon: {EPSILON:.6f}")


# 3. Load CIFAR-100
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR100(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=transform,
)

test_dataset = datasets.CIFAR100(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))
print("Number of classes:", len(train_dataset.classes))


# 4. CNN model
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)


# 5. Generate FGSM adversarial examples
def generate_fgsm(model, images, labels, epsilon):
    attack_images = images.detach().clone()
    attack_images.requires_grad_(True)

    outputs = model(attack_images)
    loss = criterion(outputs, labels)

    input_gradient = torch.autograd.grad(
        loss, attack_images
    )[0]

    adversarial_images = (
        images + epsilon * input_gradient.sign()
    ).clamp(0.0, 1.0)

    return adversarial_images.detach()


# 6. Evaluate clean accuracy
def evaluate_clean(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


# 7. Adversarial training
history = []

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
    )

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        # Generate attacks using the current model.
        adversarial_images = generate_fgsm(
            model, images, labels, EPSILON
        )

        # Train on the adversarial images.
        optimizer.zero_grad(set_to_none=True)

        outputs = model(adversarial_images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = running_loss / total
    adversarial_train_accuracy = 100.0 * correct / total
    clean_test_accuracy = evaluate_clean(model, test_loader)

    history.append({
        "epoch": epoch + 1,
        "epsilon": EPSILON,
        "training_loss": average_loss,
        "adversarial_training_accuracy":
            adversarial_train_accuracy,
        "clean_test_accuracy": clean_test_accuracy,
    })

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
        f" | Loss: {average_loss:.4f}"
        f" | Adversarial training accuracy: "
        f"{adversarial_train_accuracy:.2f}%"
        f" | Clean test accuracy: "
        f"{clean_test_accuracy:.2f}%\n"
    )


# 8. Save model and results
torch.save(model.state_dict(), CHECKPOINT_PATH)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "epsilon",
            "training_loss",
            "adversarial_training_accuracy",
            "clean_test_accuracy",
        ],
    )
    writer.writeheader()
    writer.writerows(history)

final_clean_accuracy = evaluate_clean(model, test_loader)

with open(RESULT_PATH, "w", encoding="utf-8") as file:
    file.write("CIFAR-100 FGSM Adversarial Training\n")
    file.write(f"Seed: {SEED}\n")
    file.write(f"Epochs: {EPOCHS}\n")
    file.write(f"Batch size: {BATCH_SIZE}\n")
    file.write(f"Learning rate: {LEARNING_RATE}\n")
    file.write(f"FGSM epsilon: {EPSILON:.6f}\n")
    file.write(
        f"Final clean test accuracy: "
        f"{final_clean_accuracy:.2f}%\n"
    )
    file.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
    file.write(f"Training history CSV: {CSV_PATH}\n")

print("\nCIFAR-100 adversarial training completed.")
print(f"Final clean test accuracy: {final_clean_accuracy:.2f}%")
print("Model saved to:", CHECKPOINT_PATH)
print("Results saved to:", RESULT_PATH)
print("History saved to:", CSV_PATH)
