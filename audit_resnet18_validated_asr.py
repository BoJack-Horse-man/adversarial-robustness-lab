import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.resnet_cifar import create_resnet18


# Configuration
BATCH_SIZE = 64
EPSILONS = [2 / 255, 4 / 255, 8 / 255]
PGD_ALPHA = 2 / 255
PGD_STEPS = 10

DATA_DIR = Path("data")
CHECKPOINT = Path("checkpoints/resnet18_cifar10_validated_best.pth")
OUTPUT = Path("experiments/resnet18_validated_asr_audit.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fgsm_attack(model, images, labels, epsilon):
    adversarial = images.detach().clone().requires_grad_(True)

    outputs = model(adversarial)
    loss = F.cross_entropy(outputs, labels)
    gradient = torch.autograd.grad(loss, adversarial)[0]

    return (images + epsilon * gradient.sign()).clamp(0, 1).detach()


def pgd_attack(model, images, labels, epsilon, alpha, steps):
    original = images.detach()

    noise = torch.empty_like(original).uniform_(-epsilon, epsilon)
    adversarial = (original + noise).clamp(0, 1).detach()

    for _ in range(steps):
        adversarial.requires_grad_(True)

        outputs = model(adversarial)
        loss = F.cross_entropy(outputs, labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]

        adversarial = adversarial.detach() + alpha * gradient.sign()

        perturbation = torch.clamp(
            adversarial - original,
            min=-epsilon,
            max=epsilon
        )

        adversarial = (original + perturbation).clamp(0, 1).detach()

    return adversarial


def audit_attack(model, loader, attack_name, epsilon):
    clean_correct = 0
    adversarial_correct = 0
    successful_attacks = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"{attack_name} eps={epsilon:.5f}"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)

        clean_mask = clean_predictions.eq(labels)

        if attack_name == "FGSM":
            adversarial = fgsm_attack(model, images, labels, epsilon)
        else:
            adversarial = pgd_attack(
                model, images, labels,
                epsilon, PGD_ALPHA, PGD_STEPS
            )

        with torch.no_grad():
            adversarial_predictions = model(adversarial).argmax(dim=1)

        adversarial_mask = adversarial_predictions.eq(labels)

        clean_correct += clean_mask.sum().item()
        adversarial_correct += adversarial_mask.sum().item()

        # Count only initially correct samples that become incorrect.
        successful_attacks += (
            clean_mask & ~adversarial_mask
        ).sum().item()

        total += labels.size(0)

    clean_accuracy = 100 * clean_correct / total
    adversarial_accuracy = 100 * adversarial_correct / total
    attack_success_rate = (
        100 * successful_attacks / clean_correct
        if clean_correct else 0.0
    )

    return {
        "attack": attack_name,
        "epsilon": epsilon,
        "alpha": PGD_ALPHA if attack_name == "PGD" else "",
        "steps": PGD_STEPS if attack_name == "PGD" else 1,
        "total_samples": total,
        "clean_correct": clean_correct,
        "successful_attacks": successful_attacks,
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "attack_success_rate": attack_success_rate,
    }


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = create_resnet18().to(DEVICE)
    state = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Test samples: {len(dataset)}")
    print("Starting independent ASR audit...")

    results = []

    for epsilon in EPSILONS:
        results.append(
            audit_attack(model, loader, "FGSM", epsilon)
        )

    results.append(
        audit_attack(model, loader, "PGD", 8 / 255)
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nAudit results:")
    for result in results:
        print(
            f"{result['attack']:4} | "
            f"epsilon={result['epsilon']:.5f} | "
            f"clean={result['clean_accuracy']:.2f}% | "
            f"adversarial={result['adversarial_accuracy']:.2f}% | "
            f"ASR={result['attack_success_rate']:.2f}%"
        )

    print(f"\nSaved audit results to: {OUTPUT}")


if __name__ == "__main__":
    main()
