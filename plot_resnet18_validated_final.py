import csv
from pathlib import Path

import matplotlib.pyplot as plt

INPUT_PATH = Path("experiments/resnet18_validated_combined_pruning_evaluation.csv")
OUTPUT_DIR = Path("experiments")

LEVELS = ["0%", "30%", "50%", "70%"]
MODELS = {
    "Standard": {
        "0%": "Validated_Control",
        "30%": "Pruned_30",
        "50%": "Pruned_50",
        "70%": "Pruned_70",
    },
    "FGSM_Adversarial": {
        "0%": "Validated_FGSM_Control",
        "30%": "Pruned_30",
        "50%": "Pruned_50",
        "70%": "Pruned_70",
    },
}

with INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))

def get_row(training, level, attack, epsilon=None):
    model = MODELS[training][level]

    matches = [
        row for row in rows
        if row["training_type"] == training
        and row["model"] == model
        and row["attack"] == attack
        and (
            epsilon is None
            or abs(float(row["epsilon"]) - epsilon) < 1e-7
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one result for {training}, {level}, {attack}, {epsilon}; "
            f"found {len(matches)}"
        )

    return matches[0]

def values(training, attack, metric, epsilon=None):
    return [
        float(get_row(training, level, attack, epsilon)[metric])
        for level in LEVELS
    ]

# Figure 1: Clean accuracy
plt.figure(figsize=(9, 6))

for training, label in [
    ("Standard", "Standard training"),
    ("FGSM_Adversarial", "FGSM adversarial training"),
]:
    clean = [
        float(get_row(training, level, "FGSM", 8 / 255)["clean_accuracy"])
        for level in LEVELS
    ]
    plt.plot(LEVELS, clean, marker="o", linewidth=2, label=label)

plt.xlabel("Pruning level")
plt.ylabel("Clean test accuracy (%)")
plt.title("ResNet-18 Clean Accuracy vs. Pruning Level")
plt.ylim(0, 100)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "resnet18_validated_final_clean_accuracy.png", dpi=300)
plt.close()

# Figure 2: Adversarial accuracy
plt.figure(figsize=(10, 6))

series = [
    ("Standard", "FGSM", "FGSM-8/255", 8 / 255),
    ("Standard", "PGD", "PGD-10", None),
    ("FGSM_Adversarial", "FGSM", "FGSM-8/255 (adv-trained)", 8 / 255),
    ("FGSM_Adversarial", "PGD", "PGD-10 (adv-trained)", None),
]

for training, attack, label, epsilon in series:
    accuracy = values(training, attack, "adversarial_accuracy", epsilon)
    plt.plot(LEVELS, accuracy, marker="o", linewidth=2, label=label)

plt.xlabel("Pruning level")
plt.ylabel("Adversarial test accuracy (%)")
plt.title("ResNet-18 Adversarial Accuracy vs. Pruning Level")
plt.ylim(0, 100)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "resnet18_validated_final_adversarial_accuracy.png", dpi=300)
plt.close()

# Figure 3: PGD-10 attack success rate
plt.figure(figsize=(9, 6))

for training, label in [
    ("Standard", "Standard training"),
    ("FGSM_Adversarial", "FGSM adversarial training"),
]:
    asr = values(training, "PGD", "attack_success_rate")
    plt.plot(LEVELS, asr, marker="o", linewidth=2, label=label)

plt.xlabel("Pruning level")
plt.ylabel("PGD-10 attack success rate (%)")
plt.title("ResNet-18 PGD-10 ASR vs. Pruning Level")
plt.ylim(0, 105)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "resnet18_validated_final_pgd10_asr.png", dpi=300)
plt.close()

print("All three final figures generated successfully.")

for filename in [
    "resnet18_validated_final_clean_accuracy.png",
    "resnet18_validated_final_adversarial_accuracy.png",
    "resnet18_validated_final_pgd10_asr.png",
]:
    path = OUTPUT_DIR / filename
    print(f"{path} | {path.stat().st_size:,} bytes")
