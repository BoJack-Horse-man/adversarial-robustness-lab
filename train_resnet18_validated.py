
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from models.resnet_cifar import create_resnet18


# Configuration
SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.0005
VALIDATION_SIZE = 5000

DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
RESULT_DIR = Path("experiments")

LATEST_CHECKPOINT = CHECKPOINT_DIR / "resnet18_cifar10_validated.pth"
BEST_CHECKPOINT = CHECKPOINT_DIR / "resnet18_cifar10_validated_best.pth"
CSV_PATH = RESULT_DIR / "resnet18_validated_training_history.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    transform = transforms.ToTensor()

    full_train_dataset = datasets.CIFAR10(
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

    # Reproducible 45,000 / 5,000 split.
    split_generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(
        len(full_train_dataset),
        generator=split_generator,
    ).tolist()

    validation_indices = indices[:VALIDATION_SIZE]
    training_indices = indices[VALIDATION_SIZE:]

    train_dataset = Subset(full_train_dataset, training_indices)
    validation_dataset = Subset(full_train_dataset, validation_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    validation_loader = DataLoader(
        validation_dataset,
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
    print("Validation images:", len(validation_dataset))
    print("Test images:", len(test_dataset))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Seed:", SEED)

    best_validation_accuracy = 0.0

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
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
                correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()
                total += labels.size(0)

                progress.set_postfix(
                    loss=f"{loss.item():.3f}",
                    accuracy=f"{100 * correct / total:.1f}%",
                )

            train_loss = running_loss / total
            train_accuracy = 100.0 * correct / total

            validation_loss, validation_accuracy = evaluate(
                model,
                validation_loader,
                criterion,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            print(
                f"\nEpoch {epoch + 1}/{EPOCHS} | "
                f"LR: {current_lr:.5f} | "
                f"Loss: {train_loss:.4f} | "
                f"Train: {train_accuracy:.2f}% | "
                f"Val loss: {validation_loss:.4f} | "
                f"Val: {validation_accuracy:.2f}%"
            )

            writer.writerow([
                epoch + 1,
                current_lr,
                train_loss,
                train_accuracy,
                validation_loss,
                validation_accuracy,
            ])
            file.flush()

            # Select the best checkpoint using validation accuracy only.
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(model.state_dict(), BEST_CHECKPOINT)
                print("New best validation checkpoint saved.")

            # Save the latest checkpoint separately.
            torch.save(model.state_dict(), LATEST_CHECKPOINT)

    print("\nTraining completed.")
    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.2f}%"
    )
    print("Latest checkpoint:", LATEST_CHECKPOINT)
    print("Best checkpoint:", BEST_CHECKPOINT)
    print("Training history:", CSV_PATH)


if __name__ == "__main__":
    main()
