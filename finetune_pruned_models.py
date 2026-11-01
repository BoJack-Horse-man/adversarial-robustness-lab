
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

from eval_pgd import SimpleCNN


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.0001
EPSILON = 8 / 255
VALIDATION_SIZE = 5000

DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
EXPERIMENT_DIR = Path("experiments")

MODELS = {
    "30": CHECKPOINT_DIR / "simple_cnn_adv_pruned_30.pth",
    "50": CHECKPOINT_DIR / "simple_cnn_adv_pruned_50.pth",
    "70": CHECKPOINT_DIR / "simple_cnn_adv_pruned_70.pth",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR.mkdir(exist_ok=True)
EXPERIMENT_DIR.mkdir(exist_ok=True)

HISTORY_PATH = EXPERIMENT_DIR / "pruned_finetuning_history.csv"


# --------------------------------------------------
# 2. Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------
# 3. Dataset and validation split
# --------------------------------------------------

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

split_generator = torch.Generator().manual_seed(SEED)

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [len(full_train_dataset) - VALIDATION_SIZE, VALIDATION_SIZE],
    generator=split_generator,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(f"Device: {DEVICE}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(validation_dataset)}")
print(f"Test images reserved for final evaluation: {len(test_dataset)}")


# --------------------------------------------------
# 4. FGSM generation
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


def generate_fgsm(model, images, labels):
    attack_images = images.detach().clone()
    attack_images.requires_grad_(True)

    outputs = model(attack_images)
    loss = criterion(outputs, labels)

    gradient = torch.autograd.grad(loss, attack_images)[0]

    adversarial_images = (
        images + EPSILON * gradient.sign()
    ).clamp(0.0, 1.0)

    return adversarial_images.detach()


# --------------------------------------------------
# 5. Validation accuracy
# --------------------------------------------------

def evaluate_clean(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(images).argmax(dim=1)

            correct += predictions.eq(labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


# --------------------------------------------------
# 6. Fine-tune while preserving zero weights
# --------------------------------------------------

def fine_tune_model(model_name, source_path):
    if not source_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {source_path}"
        )

    print(f"\n{'=' * 50}")
    print(f"Fine-tuning {model_name}% pruned model")

    model = SimpleCNN().to(DEVICE)

    state = torch.load(
        source_path,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state)

    # Record the original zero-weight masks.
    masks = {}

    for name, parameter in model.named_parameters():
        if parameter.ndim >= 2:
            masks[name] = parameter.detach().ne(0).float()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"{model_name}% Epoch {epoch + 1}/{EPOCHS}",
        )

        for images, labels in progress:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Generate adversarial training examples.
            adversarial_images = generate_fgsm(
                model,
                images,
                labels,
            )

            optimizer.zero_grad(set_to_none=True)

            outputs = model(adversarial_images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Enforce the original pruning masks.
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in masks:
                        parameter.mul_(masks[name])

            running_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += predictions.eq(labels).sum().item()
            total += labels.size(0)

            progress.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = running_loss / total
        training_accuracy = 100.0 * correct / total
        validation_accuracy = evaluate_clean(
            model,
            validation_loader,
        )

        row = {
            "model": model_name,
            "epoch": epoch + 1,
            "epsilon": EPSILON,
            "learning_rate": LEARNING_RATE,
            "training_loss": average_loss,
            "adversarial_training_accuracy": training_accuracy,
            "validation_clean_accuracy": validation_accuracy,
        }

        history.append(row)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
            f" | Loss: {average_loss:.4f}"
            f" | Training accuracy: {training_accuracy:.2f}%"
            f" | Validation accuracy: {validation_accuracy:.2f}%"
        )

    # Verify that all original zero weights remain zero.
    preserved = True

    for name, parameter in model.named_parameters():
        if name in masks:
            if torch.count_nonzero(parameter[masks[name] == 0]).item() != 0:
                preserved = False
                break

    if not preserved:
        raise RuntimeError(
            f"Pruning mask was not preserved for model {model_name}%"
        )

    output_path = (
        CHECKPOINT_DIR /
        f"simple_cnn_adv_pruned_{model_name}_finetuned.pth"
    )

    torch.save(model.state_dict(), output_path)

    print(f"Fine-tuned checkpoint saved: {output_path}")
    print("Original pruning zeros preserved: Yes")

    return history


# --------------------------------------------------
# 7. Run all three experiments
# --------------------------------------------------

all_history = []

for model_name, source_path in MODELS.items():
    model_history = fine_tune_model(model_name, source_path)
    all_history.extend(model_history)


# --------------------------------------------------
# 8. Save training history
# --------------------------------------------------

with HISTORY_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(all_history[0].keys()),
    )
    writer.writeheader()
    writer.writerows(all_history)

print("\nAll fine-tuning experiments completed.")
print(f"Training history saved to: {HISTORY_PATH}")
