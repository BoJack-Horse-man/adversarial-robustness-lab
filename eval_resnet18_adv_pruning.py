
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.resnet_cifar import create_resnet18


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 64

EPSILONS = [2 / 255, 4 / 255, 8 / 255]
PGD_ALPHA = 2 / 255
PGD_STEPS = 10

DATA_DIR = Path("data")
RESULT_DIR = Path("experiments")

MODELS = {
    "Control": Path("checkpoints/resnet18_cifar10_fgsm_adv.pth"),
    "Pruned_30": Path("checkpoints/resnet18_cifar10_fgsm_adv_pruned_30.pth"),
    "Pruned_50": Path("checkpoints/resnet18_cifar10_fgsm_adv_pruned_50.pth"),
    "Pruned_70": Path("checkpoints/resnet18_cifar10_fgsm_adv_pruned_70.pth"),
}

RESULT_PATH = RESULT_DIR / "resnet18_adv_pruning_comparison.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# FGSM attack
# --------------------------------------------------

def fgsm_attack(model, images, labels, epsilon):
    attack_images = images.detach().clone()
    attack_images.requires_grad_(True)

    outputs = model(attack_images)
    loss = F.cross_entropy(outputs, labels)

    gradient = torch.autograd.grad(loss, attack_images)[0]

    adversarial = (
        images + epsilon * gradient.sign()
    ).clamp(0.0, 1.0)

    return adversarial.detach()


# --------------------------------------------------
# PGD-10 attack
# --------------------------------------------------

def pgd_attack(model, images, labels, epsilon, alpha, steps):
    original = images.detach()

    noise = torch.empty_like(original).uniform_(
        -epsilon, epsilon
    )

    adversarial = (
        original + noise
    ).clamp(0.0, 1.0).detach()

    for _ in range(steps):
        adversarial.requires_grad_(True)

        outputs = model(adversarial)
        loss = F.cross_entropy(outputs, labels)

        gradient = torch.autograd.grad(
            loss, adversarial
        )[0]

        adversarial = (
            adversarial.detach() + alpha * gradient.sign()
        )

        perturbation = torch.clamp(
            adversarial - original,
            min=-epsilon,
            max=epsilon,
        )

        adversarial = (
            original + perturbation
        ).clamp(0.0, 1.0).detach()

    return adversarial


# --------------------------------------------------
# Evaluate one attack
# --------------------------------------------------

def evaluate_attack(model, loader, attack_name, epsilon):
    model.eval()

    clean_correct = 0
    adversarial_correct = 0
    attack_success = 0
    total = 0

    for images, labels in tqdm(
        loader,
        desc=f"{attack_name} {epsilon * 255:.0f}/255",
    ):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)

        clean_mask = clean_predictions.eq(labels)
        clean_correct += clean_mask.sum().item()

        if attack_name == "FGSM":
            adversarial = fgsm_attack(
                model, images, labels, epsilon
            )
        else:
            adversarial = pgd_attack(
                model,
                images,
                labels,
                epsilon,
                PGD_ALPHA,
                PGD_STEPS,
            )

        with torch.no_grad():
            adversarial_predictions = (
                model(adversarial).argmax(dim=1)
            )

        adversarial_mask = adversarial_predictions.eq(labels)

        adversarial_correct += adversarial_mask.sum().item()

        attack_success += (
            clean_mask & ~adversarial_mask
        ).sum().item()

        total += labels.size(0)

    clean_accuracy = 100.0 * clean_correct / total
    adversarial_accuracy = 100.0 * adversarial_correct / total

    asr = (
        100.0 * attack_success / clean_correct
        if clean_correct > 0 else 0.0
    )

    return clean_accuracy, adversarial_accuracy, asr


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    RESULT_DIR.mkdir(exist_ok=True)

    for name, checkpoint in MODELS.items():
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"{name} checkpoint not found: {checkpoint}"
            )

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

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Test images:", len(dataset))
    print("Batch size:", BATCH_SIZE)
    print("PGD steps:", PGD_STEPS)

    results = []

    for model_name, checkpoint in MODELS.items():
        print("\n" + "=" * 60)
        print("Evaluating:", model_name)
        print("Checkpoint:", checkpoint)
        print("=" * 60)

        model = create_resnet18().to(DEVICE)

        state = torch.load(
            checkpoint,
            map_location=DEVICE,
            weights_only=True,
        )

        model.load_state_dict(state)
        model.eval()

        for epsilon in EPSILONS:
            for attack_name in ["FGSM", "PGD"]:
                if attack_name == "PGD" and epsilon != 8 / 255:
                    continue

                set_seed(SEED)

                clean_acc, adv_acc, asr = evaluate_attack(
                    model,
                    loader,
                    attack_name,
                    epsilon,
                )

                print(
                    f"\n{model_name} | {attack_name} | "
                    f"{epsilon * 255:.0f}/255"
                )
                print(f"Clean accuracy: {clean_acc:.2f}%")
                print(f"Adversarial accuracy: {adv_acc:.2f}%")
                print(f"ASR: {asr:.2f}%")

                results.append({
                    "model": model_name,
                    "attack": attack_name,
                    "epsilon": epsilon,
                    "alpha": (
                        PGD_ALPHA if attack_name == "PGD" else ""
                    ),
                    "steps": (
                        PGD_STEPS if attack_name == "PGD" else 1
                    ),
                    "clean_accuracy": clean_acc,
                    "adversarial_accuracy": adv_acc,
                    "attack_success_rate": asr,
                })

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    with RESULT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("Pruning evaluation completed.")
    print("Results saved to:", RESULT_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()
