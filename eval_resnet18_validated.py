
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
#CHECKPOINT = Path("checkpoints/resnet18_cifar10_best.pth")
#RESULTS = Path("experiments/resnet18_attack_results.csv")
CHECKPOINT = Path("checkpoints/resnet18_cifar10_validated_best.pth")
RESULTS = Path("experiments/resnet18_validated_attack_results.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fgsm_attack(model, images, labels, epsilon):
    images = images.detach().requires_grad_(True)

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    gradient = torch.autograd.grad(loss, images)[0]

    adversarial = images.detach() + epsilon * gradient.sign()
    return adversarial.clamp(0, 1).detach()


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
            max=epsilon,
        )

        adversarial = (original + perturbation).clamp(0, 1).detach()

    return adversarial


def evaluate_attack(model, loader, attack_name, epsilon):
    clean_correct = 0
    attack_success = 0
    adversarial_correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"{attack_name} {epsilon * 255:.0f}/255"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)

        clean_mask = clean_predictions.eq(labels)
        clean_correct += clean_mask.sum().item()

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
        adversarial_correct += adversarial_mask.sum().item()

        # Count successful attacks only among initially correct predictions.
        attack_success += (
            clean_mask & ~adversarial_mask
        ).sum().item()

        total += labels.size(0)

    clean_accuracy = 100 * clean_correct / total
    adversarial_accuracy = 100 * adversarial_correct / total
    asr = 100 * attack_success / clean_correct if clean_correct else 0

    return clean_accuracy, adversarial_accuracy, asr


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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

    model = create_resnet18().to(DEVICE)
    state = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)
    print("Test images:", len(dataset))

    results = []

    for epsilon in EPSILONS:
        for attack_name in ["FGSM", "PGD"]:
            # PGD is evaluated only at epsilon 8/255.
            if attack_name == "PGD" and epsilon != 8 / 255:
                continue

            clean_acc, adv_acc, asr = evaluate_attack(
                model, loader, attack_name, epsilon
            )

            print(f"\n{attack_name} at {epsilon * 255:.0f}/255")
            print(f"Clean accuracy: {clean_acc:.2f}%")
            print(f"Adversarial accuracy: {adv_acc:.2f}%")
            print(f"Attack success rate: {asr:.2f}%")

            results.append([
                attack_name,
                epsilon,
                PGD_ALPHA if attack_name == "PGD" else "",
                PGD_STEPS if attack_name == "PGD" else 1,
                clean_acc,
                adv_acc,
                asr,
            ])

    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "attack",
            "epsilon",
            "alpha",
            "steps",
            "clean_accuracy",
            "adversarial_accuracy",
            "attack_success_rate",
        ])
        writer.writerows(results)

    print("\nResults saved to:", RESULTS)


if __name__ == "__main__":
    main()
