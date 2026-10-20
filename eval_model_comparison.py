
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# -------------------------
# 1. Settings
# -------------------------

BATCH_SIZE = 64
DATA_DIR = "./data"
RESULT_DIR = "./experiments"

BASELINE_CHECKPOINT = "./checkpoints/simple_cnn.pth"
ADV_CHECKPOINT = "./checkpoints/simple_cnn_adv.pth"

RESULT_CSV = os.path.join(
    RESULT_DIR, "model_robustness_comparison.csv"
)

EPSILONS = [0.0, 2 / 255, 4 / 255, 8 / 255]
PGD_EPSILON = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 10

os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")


# -------------------------
# 2. Model architecture
# -------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_model(checkpoint_path):
    model = SimpleCNN().to(device)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


# -------------------------
# 3. CIFAR-10 test dataset
# -------------------------

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(f"Test images: {len(test_dataset)}")


# -------------------------
# 4. FGSM attack
# -------------------------

criterion = nn.CrossEntropyLoss()


def fgsm_attack(model, images, labels, epsilon):
    attack_images = images.detach().clone()
    attack_images.requires_grad_(True)

    outputs = model(attack_images)
    loss = criterion(outputs, labels)

    input_gradient = torch.autograd.grad(
        loss, attack_images
    )[0]

    adversarial_images = (
        images + epsilon * input_gradient.sign()
    ).clamp(0.0, 1.0)

    return adversarial_images.detach()


# -------------------------
# 5. PGD attack
# -------------------------

def pgd_attack(
    model, images, labels, epsilon, alpha, steps
):
    original_images = images.detach()

    # Random start within the allowed perturbation range.
    adversarial_images = (
        original_images
        + torch.empty_like(original_images).uniform_(
            -epsilon, epsilon
        )
    ).clamp(0.0, 1.0)

    for _ in range(steps):
        adversarial_images = (
            adversarial_images.detach().requires_grad_(True)
        )

        outputs = model(adversarial_images)
        loss = criterion(outputs, labels)

        input_gradient = torch.autograd.grad(
            loss, adversarial_images
        )[0]

        adversarial_images = (
            adversarial_images.detach()
            + alpha * input_gradient.sign()
        )

        # Keep the perturbation inside the epsilon ball.
        perturbation = torch.clamp(
            adversarial_images - original_images,
            min=-epsilon,
            max=epsilon,
        )

        adversarial_images = (
            original_images + perturbation
        ).clamp(0.0, 1.0)

    return adversarial_images.detach()


# -------------------------
# 6. Evaluate accuracy
# -------------------------

def evaluate(model, attack_name, epsilon=0.0):
    correct = 0
    total = 0

    # Random-start PGD uses a random initialization.
    # Reset the RNG so both models receive the same
    # initial random perturbations for each batch.
    if attack_name == "PGD":
        torch.manual_seed(2026)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(2026)

    for images, labels in tqdm(
        test_loader,
        desc=f"{attack_name}, epsilon={epsilon:.5f}",
        leave=False,
    ):
        images = images.to(device)
        labels = labels.to(device)

        if attack_name == "Clean":
            with torch.no_grad():
                outputs = model(images)

        elif attack_name == "FGSM":
            adversarial_images = fgsm_attack(
                model, images, labels, epsilon
            )
            with torch.no_grad():
                outputs = model(adversarial_images)

        elif attack_name == "PGD":
            adversarial_images = pgd_attack(
                model,
                images,
                labels,
                PGD_EPSILON,
                PGD_ALPHA,
                PGD_STEPS,
            )
            with torch.no_grad():
                outputs = model(adversarial_images)

        else:
            raise ValueError(f"Unknown attack: {attack_name}")

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


# -------------------------
# 7. Compare both models
# -------------------------

models = {
    "Baseline": load_model(BASELINE_CHECKPOINT),
    "Adversarially trained": load_model(ADV_CHECKPOINT),
}

results = []

for model_name, model in models.items():
    print(f"\nEvaluating: {model_name}")

    clean_accuracy = evaluate(model, "Clean")

    results.append({
        "model": model_name,
        "attack": "Clean",
        "epsilon": 0.0,
        "accuracy_percent": round(clean_accuracy, 2),
    })

    print(f"Clean accuracy: {clean_accuracy:.2f}%")

    for epsilon in EPSILONS[1:]:
        accuracy = evaluate(model, "FGSM", epsilon)

        results.append({
            "model": model_name,
            "attack": "FGSM",
            "epsilon": epsilon,
            "accuracy_percent": round(accuracy, 2),
        })

        print(
            f"FGSM epsilon={epsilon:.6f}: "
            f"{accuracy:.2f}%"
        )

    pgd_accuracy = evaluate(model, "PGD")

    results.append({
        "model": model_name,
        "attack": "PGD",
        "epsilon": PGD_EPSILON,
        "accuracy_percent": round(pgd_accuracy, 2),
    })

    print(
        f"PGD epsilon={PGD_EPSILON:.6f}, "
        f"steps={PGD_STEPS}: {pgd_accuracy:.2f}%"
    )


# -------------------------
# 8. Save comparison
# -------------------------

with open(
    RESULT_CSV, "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "model",
            "attack",
            "epsilon",
            "accuracy_percent",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

print("\nComparison completed.")
print(f"Results saved to: {RESULT_CSV}")
print("\nFinal results:")

for result in results:
    print(
        f"{result['model']:24} | "
        f"{result['attack']:5} | "
        f"epsilon={result['epsilon']:.6f} | "
        f"accuracy={result['accuracy_percent']:.2f}%"
    )
