
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from models.resnet_cifar import create_resnet18


# -------------------------
# 1. Configuration
# -------------------------

SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
EPSILON = 8 / 255

DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
EXPERIMENT_DIR = Path("experiments")

BASELINE_PATH = CHECKPOINT_DIR / "resnet18_cifar10_best.pth"
CHECKPOINT_PATH = CHECKPOINT_DIR / "resnet18_cifar10_fgsm_adv.pth"
HISTORY_PATH = EXPERIMENT_DIR / "resnet18_fgsm_adv_history.csv"
RESULT_PATH = EXPERIMENT_DIR / "resnet18_fgsm_adv_result.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# 2. Reproducibility
# -------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -------------------------
# 3. FGSM attack
# -------------------------

def generate_fgsm(model, images, labels, criterion):
    attack_images = images.detach().clone()
    attack_images.requires_grad_(True)

    outputs = model(attack_images)
    loss = criterion(outputs, labels)

    input_gradient = torch.autograd.grad(
        loss, attack_images
    )[0]

    adversarial_images = (
        images + EPSILON * input_gradient.sign()
    ).clamp(0.0, 1.0)

    return adversarial_images.detach()


# -------------------------
# 4. Evaluation
# -------------------------

def evaluate_clean(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


# -------------------------
# 5. Main training
# -------------------------

def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    EXPERIMENT_DIR.mkdir(exist_ok=True)

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {BASELINE_PATH}"
        )

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

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [45000, 5000],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
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

    # Load the clean ResNet-18 baseline.
    model = create_resnet18().to(DEVICE)

    state_dict = torch.load(
        BASELINE_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[6, 8],
        gamma=0.1,
    )

    print("Starting checkpoint:", BASELINE_PATH)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("FGSM epsilon:", EPSILON)

    best_val_accuracy = -1.0
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
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # Generate adversarial examples.
            # The model is in training mode during this FGSM step.
            adversarial_images = generate_fgsm(
                model, images, labels, criterion
            )

            # Train on adversarial examples.
            optimizer.zero_grad(set_to_none=True)

            outputs = model(adversarial_images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            total += labels.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                accuracy=f"{100.0 * correct / total:.2f}%",
            )

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        val_loss, val_accuracy = evaluate_clean(
            model, val_loader, criterion
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        row = {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "epsilon": EPSILON,
            "training_loss": train_loss,
            "adversarial_training_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_clean_accuracy": val_accuracy,
        }
        history.append(row)

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
            f" | LR: {current_lr:.6f}"
            f" | Train loss: {train_loss:.4f}"
            f" | Adversarial train accuracy: {train_accuracy:.2f}%"
            f" | Validation clean accuracy: {val_accuracy:.2f}%"
        )

        # Select the checkpoint using validation accuracy only.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH,
            )
            print("New best validation checkpoint saved.")

    # Save training history.
    with HISTORY_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    # Final test evaluation using the best validation checkpoint.
    best_model = create_resnet18().to(DEVICE)
    best_state = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    best_model.load_state_dict(best_state)

    test_loss, test_accuracy = evaluate_clean(
        best_model, test_loader, criterion
    )

    with RESULT_PATH.open("w", encoding="utf-8") as file:
        file.write("CIFAR-10 ResNet-18 FGSM Adversarial Fine-Tuning\n")
        file.write(f"Seed: {SEED}\n")
        file.write(f"Starting checkpoint: {BASELINE_PATH}\n")
        file.write(f"Training images: {len(train_dataset)}\n")
        file.write(f"Validation images: {len(val_dataset)}\n")
        file.write(f"Test images: {len(test_dataset)}\n")
        file.write(f"Epochs: {EPOCHS}\n")
        file.write(f"Batch size: {BATCH_SIZE}\n")
        file.write(f"Optimizer: SGD\n")
        file.write(f"Learning rate: {LEARNING_RATE}\n")
        file.write(f"Momentum: {MOMENTUM}\n")
        file.write(f"Weight decay: {WEIGHT_DECAY}\n")
        file.write(f"FGSM epsilon: {EPSILON:.8f}\n")
        file.write(
            f"Best validation clean accuracy: "
            f"{best_val_accuracy:.2f}%\n"
        )
        file.write(f"Final test loss: {test_loss:.4f}\n")
        file.write(f"Final test clean accuracy: {test_accuracy:.2f}%\n")
        file.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        file.write(f"History: {HISTORY_PATH}\n")

    print("\nAdversarial fine-tuning completed.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("History:", HISTORY_PATH)
    print("Results:", RESULT_PATH)


if __name__ == "__main__":
    main()
