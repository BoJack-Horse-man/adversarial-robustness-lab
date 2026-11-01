import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.resnet_cifar import create_resnet18


# Configuration
SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.0005

DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
RESULT_DIR = Path("experiments")

CHECKPOINT_PATH = CHECKPOINT_DIR / "resnet18_cifar10.pth"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "resnet18_cifar10_best.pth"
CSV_PATH = RESULT_DIR / "resnet18_training_history.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = create_resnet18().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 75],
        gamma=0.1,
    )

    print("Training images:", len(train_dataset))
    print("Test images:", len(test_dataset))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    best_accuracy = 0.0

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "test_accuracy",
        ])

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
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                progress.set_postfix(
                    loss=f"{loss.item():.3f}",
                    accuracy=f"{100 * correct / total:.1f}%",
                )

            train_loss = running_loss / total
            train_accuracy = 100 * correct / total

            # Evaluate clean test accuracy.
            model.eval()
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)

                    outputs = model(images)
                    predictions = outputs.argmax(dim=1)

                    test_correct += (
                        predictions == labels
                    ).sum().item()

                    test_total += labels.size(0)

            test_accuracy = 100 * test_correct / test_total
            current_lr = optimizer.param_groups[0]["lr"]

            scheduler.step()

            print(
                f"\nEpoch {epoch + 1}/{EPOCHS} | "
                f"LR: {current_lr:.5f} | "
                f"Loss: {train_loss:.4f} | "
                f"Train: {train_accuracy:.2f}% | "
                f"Test: {test_accuracy:.2f}%"
            )

            writer.writerow([
                epoch + 1,
                current_lr,
                train_loss,
                train_accuracy,
                test_accuracy,
            ])
            file.flush()

            # Save the best model based on clean test accuracy.
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)
                print("New best checkpoint saved.")

            # Save the latest checkpoint.
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    print("\nTraining completed.")
    print(f"Best clean test accuracy: {best_accuracy:.2f}%")
    print(f"Latest model: {CHECKPOINT_PATH}")
    print(f"Best model: {BEST_CHECKPOINT_PATH}")
    print(f"Training history: {CSV_PATH}")


if __name__ == "__main__":
    main()