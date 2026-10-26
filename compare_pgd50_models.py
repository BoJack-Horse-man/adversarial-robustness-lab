
"""Compare baseline and FGSM-trained CNNs under multi-restart PGD-50."""

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from eval_pgd import SimpleCNN

SEED = 42
BATCH_SIZE = 64
EPSILON = 8 / 255
ALPHA = 2 / 255
PGD_STEPS = 50
RESTARTS = 5

DATA_DIR = Path("data")
RESULT_PATH = Path("experiments/pgd50_5restarts_model_comparison.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "baseline": Path("checkpoints/simple_cnn.pth"),
    "fgsm_adversarially_trained": Path("checkpoints/simple_cnn_adv.pth"),
}


def pgd_attack(model, images, labels):
    original = images.detach()
    adversarial = (
        original + torch.empty_like(original).uniform_(-EPSILON, EPSILON)
    ).clamp(0, 1).detach()

    for _ in range(PGD_STEPS):
        adversarial.requires_grad_(True)
        loss = nn.functional.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]

        adversarial = adversarial.detach() + ALPHA * gradient.sign()
        delta = (adversarial - original).clamp(-EPSILON, EPSILON)
        adversarial = (original + delta).clamp(0, 1).detach()

    return adversarial


def evaluate_model(name, checkpoint, loader, total_images):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = SimpleCNN().to(DEVICE)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    clean_correct = 0
    robust_correct = 0
    processed = 0

    # Reset attack randomness so both models use the same random stream.
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"\nEvaluating: {name}")
    for images, labels in tqdm(loader, desc=name):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)
        clean_correct += clean_predictions.eq(labels).sum().item()

        # A sample is robust only if all five attacks fail to fool it.
        robust_for_batch = torch.ones(
            labels.size(0), dtype=torch.bool, device=DEVICE
        )

        for _ in range(RESTARTS):
            adversarial = pgd_attack(model, images, labels)
            with torch.no_grad():
                predictions = model(adversarial).argmax(dim=1)
            robust_for_batch &= predictions.eq(labels)

        robust_correct += robust_for_batch.sum().item()
        processed += labels.size(0)

    if processed != total_images:
        raise RuntimeError(
            f"Expected {total_images} images, processed {processed}"
        )

    clean_accuracy = 100 * clean_correct / processed
    robust_accuracy = 100 * robust_correct / processed

    print(f"Clean accuracy: {clean_accuracy:.2f}%")
    print(f"PGD-50 robust accuracy: {robust_accuracy:.2f}%")

    return {
        "model": name,
        "checkpoint": str(checkpoint),
        "test_images": processed,
        "epsilon": EPSILON,
        "alpha": ALPHA,
        "steps": PGD_STEPS,
        "restarts": RESTARTS,
        "clean_accuracy_percent": f"{clean_accuracy:.2f}",
        "robust_accuracy_percent": f"{robust_accuracy:.2f}",
    }


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"Device: {DEVICE}")
    print(f"Test images: {len(dataset)}")
    print(f"PGD steps: {PGD_STEPS}; restarts: {RESTARTS}")
    print(f"Epsilon: {EPSILON:.8f}; step size: {ALPHA:.8f}")

    results = [
        evaluate_model(name, checkpoint, loader, len(dataset))
        for name, checkpoint in MODELS.items()
    ]

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n--- Model Comparison ---")
    for result in results:
        print(
            f"{result['model']}: "
            f"clean={result['clean_accuracy_percent']}%, "
            f"PGD-50={result['robust_accuracy_percent']}%"
        )
    print(f"\nSaved results to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
