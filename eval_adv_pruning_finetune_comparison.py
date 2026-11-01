
"""Compare original and fine-tuned adversarially trained pruned SimpleCNN models."""

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from eval_pgd import SimpleCNN, pgd_attack


# --------------------------------------------------
# Experiment settings
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 64

EPSILON = 8 / 255
ALPHA = 2 / 255
PGD_STEPS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data")
RESULT_PATH = Path(
    "experiments/adv_pruning_finetuning_comparison.csv"
)

MODELS = {
    "control_0": Path("checkpoints/simple_cnn_adv.pth"),

    "pruned_30": Path("checkpoints/simple_cnn_adv_pruned_30.pth"),
    "pruned_50": Path("checkpoints/simple_cnn_adv_pruned_50.pth"),
    "pruned_70": Path("checkpoints/simple_cnn_adv_pruned_70.pth"),

    "pruned_30_finetuned": Path(
        "checkpoints/simple_cnn_adv_pruned_30_finetuned.pth"
    ),
    "pruned_50_finetuned": Path(
        "checkpoints/simple_cnn_adv_pruned_50_finetuned.pth"
    ),
    "pruned_70_finetuned": Path(
        "checkpoints/simple_cnn_adv_pruned_70_finetuned.pth"
    ),
}


# --------------------------------------------------
# Measure actual sparsity
# --------------------------------------------------

def measure_sparsity(model):
    total_weights = 0
    zero_weights = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weights = module.weight.detach()

            total_weights += weights.numel()
            zero_weights += (weights == 0).sum().item()

    return 100.0 * zero_weights / total_weights


# --------------------------------------------------
# Evaluate clean and adversarial accuracy
# --------------------------------------------------

def evaluate_model(model, test_loader):
    clean_correct = 0
    adversarial_correct = 0
    successful_attacks = 0
    total_clean_correct = 0
    total = 0

    model.eval()

    for images, labels in tqdm(
        test_loader,
        desc="Evaluating",
        leave=False,
    ):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clean predictions
        with torch.no_grad():
            clean_outputs = model(images)
            clean_predictions = clean_outputs.argmax(dim=1)

        clean_correct_mask = clean_predictions.eq(labels)

        clean_correct += clean_correct_mask.sum().item()
        total_clean_correct += clean_correct_mask.sum().item()

        # PGD attack
        adversarial_images = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=EPSILON,
            alpha=ALPHA,
            steps=PGD_STEPS,
        )

        with torch.no_grad():
            adversarial_outputs = model(adversarial_images)
            adversarial_predictions = adversarial_outputs.argmax(dim=1)

        adversarial_correct_mask = adversarial_predictions.eq(labels)

        adversarial_correct += adversarial_correct_mask.sum().item()

        successful_attacks += (
            clean_correct_mask & ~adversarial_correct_mask
        ).sum().item()

        total += labels.size(0)

    clean_accuracy = 100.0 * clean_correct / total
    adversarial_accuracy = 100.0 * adversarial_correct / total

    attack_success_rate = (
        100.0 * successful_attacks / total_clean_correct
        if total_clean_correct > 0
        else 0.0
    )

    return clean_accuracy, adversarial_accuracy, attack_success_rate


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():
    transform = transforms.ToTensor()

    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"Device: {DEVICE}")
    print(f"Test images: {len(test_dataset)}")
    print(f"PGD epsilon: {EPSILON * 255:.0f}/255")
    print(f"PGD alpha: {ALPHA * 255:.0f}/255")
    print(f"PGD steps: {PGD_STEPS}")
    print("Random start: enabled")

    results = []

    for model_name, checkpoint_path in MODELS.items():

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        # Use the same random-start sequence for each model.
        torch.manual_seed(SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        print(f"\n{'=' * 50}")
        print(f"Model: {model_name}")
        print(f"Checkpoint: {checkpoint_path}")

        model = SimpleCNN().to(DEVICE)

        state = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )

        model.load_state_dict(state)
        model.eval()

        actual_sparsity = measure_sparsity(model)

        clean_acc, pgd_acc, asr = evaluate_model(
            model,
            test_loader,
        )

        result = {
            "model": model_name,
            "actual_sparsity_percent": f"{actual_sparsity:.4f}",
            "clean_accuracy_percent": f"{clean_acc:.2f}",
            "pgd10_accuracy_percent": f"{pgd_acc:.2f}",
            "attack_success_rate_percent": f"{asr:.2f}",
            "epsilon": EPSILON,
            "alpha": ALPHA,
            "steps": PGD_STEPS,
            "random_start": True,
        }

        results.append(result)

        print(f"Actual sparsity: {actual_sparsity:.4f}%")
        print(f"Clean accuracy: {clean_acc:.2f}%")
        print(f"PGD-10 accuracy: {pgd_acc:.2f}%")
        print(f"Attack success rate: {asr:.2f}%")

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Save results to a new CSV.
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nComparison evaluation completed.")
    print(f"Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
