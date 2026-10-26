
"""Compare clean CIFAR-10 accuracy before and after 50% pruning."""

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from eval_pgd import SimpleCNN

SEED = 42
BATCH_SIZE = 64
DATA_DIR = Path("data")
RESULT_PATH = Path("experiments/pruning_clean_comparison.csv")

CHECKPOINTS = {
    "baseline": Path("checkpoints/simple_cnn.pth"),
    "pruned_50_percent": Path("checkpoints/simple_cnn_pruned_50.pth"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_clean(model, loader):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def main():
    torch.manual_seed(SEED)

    for name, path in CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    results = []

    print(f"Device: {DEVICE}")
    print(f"Test images: {len(test_dataset)}")
    print("Evaluating both checkpoints on the same test set.\n")

    for name, checkpoint_path in CHECKPOINTS.items():
        model = SimpleCNN().to(DEVICE)

        state_dict = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
        model.load_state_dict(state_dict)

        accuracy = evaluate_clean(model, test_loader)

        results.append({
            "model": name,
            "checkpoint": str(checkpoint_path),
            "test_images": len(test_dataset),
            "clean_accuracy_percent": f"{accuracy:.2f}",
        })

        print(f"{name}: {accuracy:.2f}%")

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    baseline_accuracy = float(results[0]["clean_accuracy_percent"])
    pruned_accuracy = float(results[1]["clean_accuracy_percent"])
    difference = pruned_accuracy - baseline_accuracy

    print(f"\nClean accuracy change: {difference:+.2f} percentage points")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to {RESULT_PATH}")


if __name__ == "__main__":
    main()
