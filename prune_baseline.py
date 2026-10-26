
"""Apply 50% global magnitude pruning to the baseline CIFAR-10 CNN."""

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from eval_pgd import SimpleCNN

SEED = 42
SPARSITY = 0.50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOURCE_PATH = Path("checkpoints/simple_cnn.pth")
OUTPUT_PATH = Path("checkpoints/simple_cnn_pruned_50.pth")
RESULT_PATH = Path("experiments/pruning_baseline_results.csv")


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def measure_weight_sparsity(model):
    total_weights = 0
    zero_weights = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weights = module.weight.detach()
            total_weights += weights.numel()
            zero_weights += (weights == 0).sum().item()

    return zero_weights, total_weights, 100 * zero_weights / total_weights


def main():
    torch.manual_seed(SEED)

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {SOURCE_PATH}")

    model = SimpleCNN().to(DEVICE)
    state = torch.load(
        SOURCE_PATH, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    original_parameters = count_parameters(model)
    original_size_bytes = SOURCE_PATH.stat().st_size

    # Prune all convolutional and linear weights together.
    parameters_to_prune = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=SPARSITY,
    )

    # Permanently apply masks so the saved model has ordinary zero weights.
    for module, parameter_name in parameters_to_prune:
        prune.remove(module, parameter_name)

    zero_weights, total_weights, actual_sparsity = (
        measure_weight_sparsity(model)
    )

    # Save a plain state_dict at a new path; keep the source checkpoint intact.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_PATH)

    result = {
        "model": "baseline_global_magnitude_pruned",
        "target_sparsity_percent": f"{SPARSITY * 100:.2f}",
        "actual_weight_sparsity_percent": f"{actual_sparsity:.2f}",
        "zero_weights": zero_weights,
        "total_prunable_weights": total_weights,
        "total_parameters": original_parameters,
        "source_checkpoint_bytes": original_size_bytes,
        "pruned_checkpoint_bytes": OUTPUT_PATH.stat().st_size,
        "device": str(DEVICE),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULT_PATH.exists()
    with RESULT_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    print("\n--- Baseline Pruning Results ---")
    print(f"Device: {DEVICE}")
    print(f"Total model parameters: {original_parameters:,}")
    print(f"Target sparsity: {SPARSITY * 100:.1f}%")
    print(f"Actual zero weights: {zero_weights:,}/{total_weights:,}")
    print(f"Actual weight sparsity: {actual_sparsity:.2f}%")
    print(f"Original checkpoint: {original_size_bytes:,} bytes")
    print(f"Pruned checkpoint: {OUTPUT_PATH.stat().st_size:,} bytes")
    print(f"Saved checkpoint: {OUTPUT_PATH}")
    print(f"Saved results: {RESULT_PATH}")


if __name__ == "__main__":
    main()
