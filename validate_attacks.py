
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import torchattacks
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.resnet_cifar import create_resnet18


# Configuration
SEED = 42
BATCH_SIZE = 64

EPSILONS = [2 / 255, 4 / 255, 8 / 255]
PGD_EPSILON = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 10

DATA_DIR = Path("data")
CHECKPOINT = Path("checkpoints/resnet18_cifar10_best.pth")
RESULT_PATH = Path("experiments/attack_implementation_validation.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_fgsm(model, images, labels, epsilon):
    images = images.detach().requires_grad_(True)

    loss = F.cross_entropy(model(images), labels)
    gradient = torch.autograd.grad(loss, images)[0]

    return (images.detach() + epsilon * gradient.sign()).clamp(0, 1)


def custom_pgd(model, images, labels, epsilon, alpha, steps):
    original = images.detach()

    noise = torch.empty_like(original).uniform_(-epsilon, epsilon)
    adversarial = (original + noise).clamp(0, 1).detach()

    for _ in range(steps):
        adversarial.requires_grad_(True)

        loss = F.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]

        adversarial = adversarial.detach() + alpha * gradient.sign()

        perturbation = torch.clamp(
            adversarial - original,
            min=-epsilon,
            max=epsilon,
        )

        adversarial = (original + perturbation).clamp(0, 1).detach()

    return adversarial


def evaluate(model, loader, attack_name, epsilon, implementation):
    clean_correct = 0
    adversarial_correct = 0
    successful_attacks = 0
    total = 0
    max_perturbation = 0.0

    if attack_name == "FGSM":
        if implementation == "torchattacks":
            attack = torchattacks.FGSM(model, eps=epsilon)

    elif attack_name == "PGD":
        if implementation == "torchattacks":
            attack = torchattacks.PGD(
                model,
                eps=epsilon,
                alpha=PGD_ALPHA,
                steps=PGD_STEPS,
                random_start=True,
            )

    for images, labels in tqdm(
        loader,
        desc=f"{implementation} {attack_name} {epsilon * 255:.0f}/255",
    ):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)

        clean_mask = clean_predictions.eq(labels)
        clean_correct += clean_mask.sum().item()

        if attack_name == "FGSM":
            if implementation == "custom":
                adversarial = custom_fgsm(
                    model, images, labels, epsilon
                )
            else:
                adversarial = attack(images, labels)

        else:
            if implementation == "custom":
                adversarial = custom_pgd(
                    model,
                    images,
                    labels,
                    epsilon,
                    PGD_ALPHA,
                    PGD_STEPS,
                )
            else:
                adversarial = attack(images, labels)

        perturbation = (adversarial - images).abs().max().item()
        max_perturbation = max(max_perturbation, perturbation)

        with torch.no_grad():
            adversarial_predictions = model(adversarial).argmax(dim=1)

        adversarial_mask = adversarial_predictions.eq(labels)

        adversarial_correct += adversarial_mask.sum().item()

        successful_attacks += (
            clean_mask & ~adversarial_mask
        ).sum().item()

        total += labels.size(0)

    clean_accuracy = 100 * clean_correct / total
    adversarial_accuracy = 100 * adversarial_correct / total

    asr = (
        100 * successful_attacks / clean_correct
        if clean_correct else 0
    )

    return {
        "implementation": implementation,
        "attack": attack_name,
        "epsilon": epsilon,
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "asr": asr,
        "max_perturbation": max_perturbation,
    }


def main():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

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
    print("Starting independent attack validation...")

    results = []

    # Compare both implementations under identical attack settings.
    for epsilon in EPSILONS:
        for implementation in ["custom", "torchattacks"]:
            results.append(
                evaluate(
                    model, loader, "FGSM", epsilon, implementation
                )
            )

    for implementation in ["custom", "torchattacks"]:
        results.append(
            evaluate(
                model,
                loader,
                "PGD",
                PGD_EPSILON,
                implementation,
            )
        )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "implementation",
                "attack",
                "epsilon",
                "clean_accuracy",
                "adversarial_accuracy",
                "asr",
                "max_perturbation",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n=== VALIDATION SUMMARY ===")

    for result in results:
        print(
            f"{result['implementation']:12} "
            f"{result['attack']:5} "
            f"epsilon={result['epsilon'] * 255:.0f}/255 | "
            f"clean={result['clean_accuracy']:.2f}% | "
            f"adversarial={result['adversarial_accuracy']:.2f}% | "
            f"ASR={result['asr']:.2f}% | "
            f"max perturbation={result['max_perturbation']:.6f}"
        )

    print("\nResults saved to:", RESULT_PATH)


if __name__ == "__main__":
    main()
