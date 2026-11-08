import csv
from pathlib import Path

import matplotlib.pyplot as plt

INPUT = Path("experiments/resnet18_validated_combined_pruning_evaluation.csv")
OUT = Path("experiments")

LEVELS = ["0%", "30%", "50%", "70%"]
MODELS = {
    "Standard": ["Validated_Control", "Pruned_30", "Pruned_50", "Pruned_70"],
    "FGSM_Adversarial": [
        "Validated_FGSM_Control", "Pruned_30", "Pruned_50", "Pruned_70"
    ],
}

with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

def get_value(training, index, attack, metric, epsilon=None):
    model = MODELS[training][index]
    found = [
        r for r in rows
        if r["training_type"] == training
        and r["model"] == model
        and r["attack"] == attack
        and (epsilon is None or abs(float(r["epsilon"]) - epsilon) < 1e-7)
    ]
    if len(found) != 1:
        raise ValueError(f"Unexpected result count: {training}, {model}, {attack}")
    return float(found[0][metric])

# Publication-style settings
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
})

BLUE = "#2878B5"
ORANGE = "#E07A35"
DARK = "#303846"
GRID = "#DCE2E8"

def polish(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#AAB2BC")
    ax.spines["bottom"].set_color("#AAB2BC")
    ax.tick_params(length=0, pad=7)

def annotate_points(ax, x, y, color, offset=7):
    for i, value in enumerate(y):
        ax.annotate(
            f"{value:.1f}",
            (i, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=color,
        )

# Figure 1: Clean accuracy
fig, ax = plt.subplots(figsize=(9, 5.6))

for training, label, color, marker in [
    ("Standard", "Standard training", BLUE, "o"),
    ("FGSM_Adversarial", "FGSM adversarial training", ORANGE, "s"),
]:
    y = [
        get_value(training, i, "FGSM", "clean_accuracy", 8 / 255)
        for i in range(4)
    ]
    ax.plot(
        LEVELS, y, color=color, marker=marker, markersize=7,
        linewidth=2.5, label=label,
    )
    annotate_points(ax, range(4), y, color)

ax.set_title("Clean Accuracy under Different Pruning Levels", pad=17, color=DARK)
ax.set_xlabel("Unstructured pruning level", labelpad=10)
ax.set_ylabel("Clean test accuracy (%)", labelpad=10)
ax.set_ylim(45, 100)
ax.legend(frameon=False, loc="lower left")
polish(ax)
fig.tight_layout()
fig.savefig(OUT / "resnet18_validated_final_clean_accuracy_v2.png",
            bbox_inches="tight")
plt.close(fig)

# Figure 2: Adversarial accuracy, two separate panels
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

for ax, training, title in [
    (axes[0], "Standard", "Standard training"),
    (axes[1], "FGSM_Adversarial", "FGSM adversarial training"),
]:
    for attack, label, epsilon, color, marker in [
        ("FGSM", "FGSM (8/255)", 8 / 255, BLUE, "o"),
        ("PGD", "PGD-10", None, ORANGE, "s"),
    ]:
        y = [
            get_value(training, i, attack, "adversarial_accuracy", epsilon)
            for i in range(4)
        ]
        ax.plot(
            LEVELS, y, color=color, marker=marker,
            markersize=7, linewidth=2.5, label=label,
        )
        annotate_points(ax, range(4), y, color)

    ax.set_title(title, pad=14, color=DARK)
    ax.set_xlabel("Pruning level", labelpad=9)
    ax.set_ylim(-3, 48)
    ax.legend(frameon=False, loc="upper right")
    polish(ax)

axes[0].set_ylabel("Adversarial test accuracy (%)", labelpad=10)
fig.suptitle("Adversarial Accuracy after Pruning", fontsize=15,
             fontweight="bold", color=DARK, y=1.02)
fig.text(0.5, -0.015, "PGD-10: ε = 8/255, α = 2/255, 10 steps",
         ha="center", fontsize=9, color="#626B76")
fig.tight_layout()
fig.savefig(OUT / "resnet18_validated_final_adversarial_accuracy_v2.png",
            bbox_inches="tight")
plt.close(fig)

# Figure 3: PGD-10 attack success rate
fig, ax = plt.subplots(figsize=(9, 5.6))

for training, label, color, marker in [
    ("Standard", "Standard training", BLUE, "o"),
    ("FGSM_Adversarial", "FGSM adversarial training", ORANGE, "s"),
]:
    y = [
        get_value(training, i, "PGD", "attack_success_rate")
        for i in range(4)
    ]
    ax.plot(
        LEVELS, y, color=color, marker=marker,
        markersize=7, linewidth=2.5, label=label,
    )
    annotate_points(ax, range(4), y, color, offset=8)

ax.set_title("PGD-10 Attack Success Rate", pad=17, color=DARK)
ax.set_xlabel("Unstructured pruning level", labelpad=10)
ax.set_ylabel("Attack success rate (%)", labelpad=10)
ax.set_ylim(35, 110)
ax.legend(frameon=False, loc="lower left")
polish(ax)
fig.tight_layout()
fig.savefig(OUT / "resnet18_validated_final_pgd10_asr_v2.png",
            bbox_inches="tight")
plt.close(fig)

print("Redesigned figures generated:")
for name in [
    "resnet18_validated_final_clean_accuracy_v2.png",
    "resnet18_validated_final_adversarial_accuracy_v2.png",
    "resnet18_validated_final_pgd10_asr_v2.png",
]:
    path = OUT / name
    print(f"{name}: {path.stat().st_size:,} bytes")
