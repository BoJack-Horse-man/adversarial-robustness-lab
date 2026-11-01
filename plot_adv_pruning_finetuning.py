
import csv
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_PATH = Path("experiments/adv_pruning_finetuning_comparison.csv")
OUTPUT_DIR = Path("experiments")

with INPUT_PATH.open("r", encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))

# Keep the three original and three fine-tuned pruning models.
levels = ["30", "50", "70"]

original = {}
finetuned = {}

for row in rows:
    name = row["model"]

    if name.startswith("pruned_") and name.endswith("_finetuned"):
        level = name.split("_")[1]
        finetuned[level] = row
    elif name.startswith("pruned_"):
        level = name.split("_")[1]
        original[level] = row

if not all(level in original and level in finetuned for level in levels):
    raise ValueError("Missing pruning results in the comparison CSV.")

metrics = [
    ("clean_accuracy_percent", "Clean Accuracy (%)",
     "clean_accuracy_before_after.png"),
    ("pgd10_accuracy_percent", "PGD-10 Accuracy (%)",
     "pgd10_accuracy_before_after.png"),
    ("attack_success_rate_percent", "Attack Success Rate (%)",
     "asr_before_after.png"),
]

for column, title, filename in metrics:
    before = [float(original[level][column]) for level in levels]
    after = [float(finetuned[level][column]) for level in levels]

    positions = list(range(len(levels)))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(
        [p - width / 2 for p in positions],
        before,
        width,
        label="Before fine-tuning",
    )
    plt.bar(
        [p + width / 2 for p in positions],
        after,
        width,
        label="After fine-tuning",
    )

    plt.xticks(positions, [f"{level}% pruning" for level in levels])
    plt.ylabel(title)
    plt.title(title + " Before and After Fine-tuning")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

print("\nAll three comparison graphs generated.")