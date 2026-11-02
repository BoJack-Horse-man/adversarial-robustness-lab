
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from models.resnet_cifar import create_resnet18


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT = Path("checkpoints/resnet18_cifar10_fgsm_adv.pth")
OUTPUT_DIR = Path("checkpoints")
RESULTS_FILE = Path("experiments/resnet18_adv_pruning_sparsity.csv")

PRUNING_RATIOS = [0.30, 0.50, 0.70]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_prunable_layers(model):
    layers = []

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((module, "weight"))

    return layers


def calculate_sparsity(model):
    total_weights = 0
    zero_weights = 0

    for module, name in get_prunable_layers(model):
        weights = getattr(module, name).detach()
        total_weights += weights.numel()
        zero_weights += (weights == 0).sum().item()

    sparsity = 100.0 * zero_weights / total_weights

    return total_weights, zero_weights, sparsity


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading adversarially trained ResNet-18...")

    model = create_resnet18().to(DEVICE)

    state = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state)
    model.eval()

    layers = get_prunable_layers(model)

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    total_weights, _, original_sparsity = calculate_sparsity(model)

    print("Total parameters:", total_parameters)
    print("Prunable weights:", total_weights)
    print("Initial sparsity:", f"{original_sparsity:.4f}%")
    print("Prunable layers:", len(layers))

    results = []

    for ratio in PRUNING_RATIOS:
        print("\n" + "=" * 55)
        print(f"Applying {ratio * 100:.0f}% global magnitude pruning")
        print("=" * 55)

        # Reload the original checkpoint for each ratio.
        pruned_model = create_resnet18().to(DEVICE)
        pruned_model.load_state_dict(state)
        pruned_model.eval()

        layers = get_prunable_layers(pruned_model)

        # Global L1 unstructured pruning.
        prune.global_unstructured(
            layers,
            pruning_method=prune.L1Unstructured,
            amount=ratio,
        )

        # Remove pruning reparameterization while preserving zero weights.
        for module, name in layers:
            prune.remove(module, name)

        total_weights, zero_weights, actual_sparsity = (
            calculate_sparsity(pruned_model)
        )

        checkpoint_path = (
            OUTPUT_DIR /
            f"resnet18_cifar10_fgsm_adv_pruned_{int(ratio * 100)}.pth"
        )

        torch.save(pruned_model.state_dict(), checkpoint_path)

        print("Target sparsity:", f"{ratio * 100:.2f}%")
        print("Actual sparsity:", f"{actual_sparsity:.4f}%")
        print("Zero weights:", zero_weights)
        print("Total prunable weights:", total_weights)
        print("Checkpoint saved:", checkpoint_path)

        results.append({
            "target_sparsity": ratio * 100,
            "actual_sparsity": actual_sparsity,
            "zero_weights": zero_weights,
            "total_prunable_weights": total_weights,
            "checkpoint": str(checkpoint_path),
        })

        del pruned_model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "target_sparsity",
                "actual_sparsity",
                "zero_weights",
                "total_prunable_weights",
                "checkpoint",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nPruning completed.")
    print("Results saved to:", RESULTS_FILE)


if __name__ == "__main__":
    main()
