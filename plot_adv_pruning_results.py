
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
RESULT_PATH = Path("experiments/adv_pruning_evaluation.csv")
OUTPUT_PATH = Path("experiments/adv_pruning_accuracy.png")

if not RESULT_PATH.exists():
    raise FileNotFoundError(f"Results file not found: {RESULT_PATH}")

# Read CSV without pandas
sparsity = []
clean_accuracy = []
pgd_accuracy = []

with RESULT_PATH.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        sparsity.append(float(row["actual_sparsity_percent"]))
        clean_accuracy.append(float(row["clean_accuracy_percent"]))
        pgd_accuracy.append(float(row["pgd10_accuracy_percent"]))

# Sort by sparsity
data = sorted(zip(sparsity, clean_accuracy, pgd_accuracy))

sparsity, clean_accuracy, pgd_accuracy = zip(*data)

# Create graph
plt.figure(figsize=(9, 6))

plt.plot(
    sparsity,
    clean_accuracy,
    marker="o",
    linewidth=2,
    markersize=7,
    label="Clean Accuracy",
)

plt.plot(
    sparsity,
    pgd_accuracy,
    marker="s",
    linewidth=2,
    markersize=7,
    label="PGD-10 Accuracy",
)

plt.xlabel("Weight Sparsity (%)")
plt.ylabel("Accuracy (%)")
plt.title("Effect of Magnitude Pruning on Model Accuracy")

plt.xticks([0, 30, 50, 70])
plt.ylim(0, 55)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

# Save graph
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

print("Graph created successfully.")
print(f"Saved to: {OUTPUT_PATH}")

plt.show()
