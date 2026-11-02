import csv
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.resnet_cifar import create_resnet18


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
EPSILON = 8 / 255

PRUNING_RATIOS = [30, 50, 70]

DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
EXPERIMENT_DIR = Path("experiments")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

def get_data_loaders():
    transform = transforms.ToTensor()

    full_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    train_dataset, val_dataset = random_split(
        full_dataset,
        [45000, 5000],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

    return train_loader, val_loader


# --------------------------------------------------
# Pruning masks
# --------------------------------------------------

def get_pruning_masks(model):
    masks = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            masks[name] = (module.weight.detach() != 0).to(
                dtype=module.weight.dtype
            )

    return masks


def apply_masks_to_gradients(model, masks):
    for name, module in model.named_modules():
        if name in masks and module.weight.grad is not None:
            module.weight.grad.mul_(masks[name])


def enforce_masks(model, masks):
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in masks:
                module.weight.mul_(masks[name])


def calculate_sparsity(model):
    total_weights = 0
    zero_weights = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weights = module.weight.detach()
            total_weights += weights.numel()
            zero_weights += (weights == 0).sum().item()

    return 100.0 * zero_weights / total_weights


# --------------------------------------------------
# FGSM attack
# --------------------------------------------------

def generate_fgsm(model, images, labels):
    images = images.detach().clone().requires_grad_(True)

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    gradients = torch.autograd.grad(
        loss,
        images,
        retain_graph=False,
        create_graph=False,
    )[0]

    adversarial_images = (
        images + EPSILON * gradients.sign()
    ).clamp(0, 1)

    return adversarial_images.detach()


# --------------------------------------------------
# Validation
# --------------------------------------------------

def evaluate_clean(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


# --------------------------------------------------
# Fine-tuning one pruned model
# --------------------------------------------------

def fine_tune_model(ratio, train_loader, val_loader):
    print("\n" + "=" * 65)
    print(f"Fine-tuning the {ratio}% pruned ResNet-18")
    print("=" * 65)

    source_checkpoint = (
        CHECKPOINT_DIR /
        f"resnet18_cifar10_fgsm_adv_pruned_{ratio}.pth"
    )

    output_checkpoint = (
        CHECKPOINT_DIR /
        f"resnet18_cifar10_fgsm_adv_pruned_{ratio}_finetuned.pth"
    )

    history_file = (
        EXPERIMENT_DIR /
        f"resnet18_pruned_{ratio}_finetuning_history.csv"
    )

    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)

    model = create_resnet18().to(DEVICE)

    state = torch.load(
        source_checkpoint,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state)

    masks = get_pruning_masks(model)

    initial_sparsity = calculate_sparsity(model)

    print(f"Initial sparsity: {initial_sparsity:.6f}%")

    if abs(initial_sparsity - ratio) > 0.01:
        raise RuntimeError(
            f"Expected {ratio}% sparsity, "
            f"but found {initial_sparsity:.6f}%"
        )

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

    best_val_accuracy = 0.0
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Generate FGSM adversarial examples.
            # The model remains in training mode, matching
            # the existing ResNet FGSM fine-tuning method.
            adversarial_images = generate_fgsm(
                model,
                images,
                labels,
            )

            # Train on adversarial examples.
            optimizer.zero_grad(set_to_none=True)

            outputs = model(adversarial_images)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()

            # Keep pruned weights from receiving updates.
            apply_masks_to_gradients(model, masks)

            optimizer.step()

            # Ensure pruned weights remain exactly zero.
            enforce_masks(model, masks)

            running_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        val_accuracy = evaluate_clean(model, val_loader)
        current_sparsity = calculate_sparsity(model)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Train: {train_accuracy:.2f}% "
            f"Validation: {val_accuracy:.2f}% "
            f"Sparsity: {current_sparsity:.4f}%"
        )

        if abs(current_sparsity - ratio) > 0.01:
            raise RuntimeError(
                f"Pruning mask changed! Expected {ratio}%, "
                f"found {current_sparsity:.6f}%"
            )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "sparsity": current_sparsity,
        })

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No best checkpoint was produced.")

    torch.save(best_state, output_checkpoint)

    with history_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "validation_accuracy",
                "sparsity",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    print("\nFine-tuning completed.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Final checkpoint: {output_checkpoint}")
    print(f"Training history: {history_file}")

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "pruning_ratio": ratio,
        "best_validation_accuracy": best_val_accuracy,
        "final_sparsity": current_sparsity,
        "checkpoint": str(output_checkpoint),
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading CIFAR-10 training and validation data...")
    train_loader, val_loader = get_data_loaders()

    results = []

    for ratio in PRUNING_RATIOS:
        # Reset random state before each independent experiment.
        set_seed(SEED)

        result = fine_tune_model(
            ratio,
            train_loader,
            val_loader,
        )

        results.append(result)

    summary_file = (
        EXPERIMENT_DIR /
        "resnet18_pruned_finetuning_summary.csv"
    )

    with summary_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pruning_ratio",
                "best_validation_accuracy",
                "final_sparsity",
                "checkpoint",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print("ALL PRUNED MODEL FINE-TUNING EXPERIMENTS COMPLETED")
    print("=" * 65)

    for result in results:
        print(
            f"{result['pruning_ratio']}% pruning | "
            f"Validation: {result['best_validation_accuracy']:.2f}% | "
            f"Sparsity: {result['final_sparsity']:.4f}%"
        )

    print("Summary:", summary_file)


if __name__ == "__main__":
    main()