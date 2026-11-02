
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.resnet_cifar import create_resnet18


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 32
EPSILON = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 50
RESTARTS = 5
BASE_SEED = 42

DATA_DIR = Path("data")

MODELS = {
    "Baseline": Path("checkpoints/resnet18_cifar10_best.pth"),
    "FGSM_Adversarial": Path("checkpoints/resnet18_cifar10_fgsm_adv.pth"),
}

RESULTS = Path("experiments/resnet18_pgd50_multirestart.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# PGD attack
# --------------------------------------------------

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


# --------------------------------------------------
# Evaluation
# --------------------------------------------------


def evaluate_model(model, loader, model_name):
    total = 0
    clean_correct = 0

    # One Boolean mask per batch, combining successful attacks
    # across all random restarts.
    successful_across_restarts = []

    restart_success_counts = [0] * RESTARTS

    for restart in range(RESTARTS):
        set_seed(BASE_SEED + restart)
        restart_success = 0

        progress = tqdm(
            loader,
            desc=f"{model_name} PGD-50 restart {restart + 1}/{RESTARTS}",
        )

        for batch_number, (images, labels) in enumerate(progress):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.no_grad():
                clean_predictions = model(images).argmax(dim=1)

            clean_mask = clean_predictions.eq(labels)

            if restart == 0:
                clean_correct += clean_mask.sum().item()
                total += labels.size(0)

            adversarial = pgd_attack(
                model,
                images,
                labels,
                EPSILON,
                PGD_ALPHA,
                PGD_STEPS,
            )

            with torch.no_grad():
                adversarial_predictions = model(adversarial).argmax(dim=1)

            adversarial_mask = adversarial_predictions.eq(labels)
            success = clean_mask & ~adversarial_mask

            restart_success += success.sum().item()

            if restart == 0:
                successful_across_restarts.append(
                    success.detach().cpu()
                )
            else:
                successful_across_restarts[batch_number] |= (
                    success.detach().cpu()
                )

        restart_success_counts[restart] = restart_success

        restart_asr = (
            100 * restart_success / clean_correct
            if clean_correct else 0
        )

        print(
            f"{model_name} restart {restart + 1}: "
            f"successful attacks = {restart_success}, "
            f"ASR = {restart_asr:.2f}%"
        )

    total_success = sum(
        mask.sum().item()
        for mask in successful_across_restarts
    )

    cumulative_asr = (
        100 * total_success / clean_correct
        if clean_correct else 0
    )

    robust_correct = clean_correct - total_success
    robust_accuracy = 100 * robust_correct / total

    clean_accuracy = 100 * clean_correct / total

    return {
        "model": model_name,
        "clean_accuracy": clean_accuracy,
        "pgd50_robust_accuracy": robust_accuracy,
        "cumulative_asr": cumulative_asr,
        "clean_correct": clean_correct,
        "total_images": total,
        "restart_success_counts": restart_success_counts,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Batch size:", BATCH_SIZE)
    print("PGD epsilon:", EPSILON)
    print("PGD alpha:", PGD_ALPHA)
    print("PGD steps:", PGD_STEPS)
    print("Random restarts:", RESTARTS)

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

    print("Test images:", len(dataset))

    results = []

    for model_name, checkpoint in MODELS.items():
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

        print("\n" + "=" * 60)
        print("Evaluating:", model_name)
        print("Checkpoint:", checkpoint)
        print("=" * 60)

        set_seed(BASE_SEED)

        model = create_resnet18().to(DEVICE)

        state = torch.load(
            checkpoint,
            map_location=DEVICE,
            weights_only=True,
        )

        model.load_state_dict(state)
        model.eval()

        result = evaluate_model(model, loader, model_name)
        results.append(result)

        print("\nSummary:", model_name)
        print(f"Clean accuracy: {result['clean_accuracy']:.2f}%")
        print(
            "PGD-50 robust accuracy: "
            f"{result['pgd50_robust_accuracy']:.2f}%"
        )
        print(f"Cumulative ASR: {result['cumulative_asr']:.2f}%")

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "model",
            "epsilon",
            "alpha",
            "steps",
            "restarts",
            "clean_accuracy",
            "pgd50_robust_accuracy",
            "cumulative_asr",
            "restart_success_counts",
        ])

        for result in results:
            writer.writerow([
                result["model"],
                EPSILON,
                PGD_ALPHA,
                PGD_STEPS,
                RESTARTS,
                result["clean_accuracy"],
                result["pgd50_robust_accuracy"],
                result["cumulative_asr"],
                ";".join(map(str, result["restart_success_counts"])),
            ])

    print("\n" + "=" * 60)
    print("Evaluation completed.")
    print("Results saved to:", RESULTS)
    print("=" * 60)


if __name__ == "__main__":
    main()
