
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from eval_pgd import SimpleCNN


# Experiment settings
SEED = 42
SPARSITY_LEVELS = [0.30, 0.50, 0.70]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOURCE_PATH = Path("checkpoints/simple_cnn_adv.pth")
OUTPUT_DIR = Path("checkpoints")
RESULT_PATH = Path("experiments/adv_pruning_comparison.csv")


def measure_sparsity(model):
    total_weights = 0
    zero_weights = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weights = module.weight.detach()
            total_weights += weights.numel()
            zero_weights += (weights == 0).sum().item()

    actual_sparsity = 100.0 * zero_weights / total_weights

    return zero_weights, total_weights, actual_sparsity


def load_model():
    model = SimpleCNN().to(DEVICE)

    state = torch.load(
        SOURCE_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    return model


def main():
    torch.manual_seed(SEED)

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Source checkpoint not found: {SOURCE_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []

    # First, record the unpruned adversarially trained control.
    model = load_model()

    zero, total, actual = measure_sparsity(model)

    results.append({
        "model": "adv_trained_control",
        "target_sparsity_percent": 0,
        "actual_sparsity_percent": f"{actual:.4f}",
        "zero_weights": zero,
        "total_prunable_weights": total,
        "checkpoint": str(SOURCE_PATH),
        "checkpoint_bytes": SOURCE_PATH.stat().st_size,
    })

    print(f"Device: {DEVICE}")
    print(f"Control model actual sparsity: {actual:.4f}%")

    # Each pruning level starts from the original checkpoint.
    for sparsity in SPARSITY_LEVELS:
        model = load_model()

        parameters_to_prune = [
            (module, "weight")
            for module in model.modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )

        # Permanently apply masks.
        for module, parameter_name in parameters_to_prune:
            prune.remove(module, parameter_name)

        zero, total, actual = measure_sparsity(model)

        percentage = int(sparsity * 100)

        output_path = (
            OUTPUT_DIR /
            f"simple_cnn_adv_pruned_{percentage}.pth"
        )

        torch.save(model.state_dict(), output_path)

        result = {
            "model": f"adv_trained_pruned_{percentage}",
            "target_sparsity_percent": percentage,
            "actual_sparsity_percent": f"{actual:.4f}",
            "zero_weights": zero,
            "total_prunable_weights": total,
            "checkpoint": str(output_path),
            "checkpoint_bytes": output_path.stat().st_size,
        }

        results.append(result)

        print(f"\nPruning level: {percentage}%")
        print(f"Actual sparsity: {actual:.4f}%")
        print(f"Zero weights: {zero:,}/{total:,}")
        print(f"Checkpoint saved: {output_path}")

    # Save all results to a separate CSV.
    fieldnames = list(results[0].keys())

    with RESULT_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nPruning experiment completed.")
    print(f"Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
