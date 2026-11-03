
import csv
import random
import statistics
import time
from pathlib import Path

import torch

from models.resnet_cifar import create_resnet18


# ---------------- Configuration ----------------

BATCH_SIZES = [1, 32, 64]
WARMUP_STEPS = 20
TIMED_STEPS = 100
ROUNDS = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_PATH = Path(
    "experiments/resnet18_efficiency_v2_results.csv"
)

MODELS = {
    "Baseline": Path("checkpoints/resnet18_cifar10_best.pth"),
    "FGSM_Adversarial": Path("checkpoints/resnet18_cifar10_fgsm_adv.pth"),
    "Pruned_30": Path("checkpoints/resnet18_cifar10_fgsm_adv_pruned_30.pth"),
    "Pruned_50": Path("checkpoints/resnet18_cifar10_fgsm_adv_pruned_50.pth"),
    "Pruned_70": Path("checkpoints/resnet18_cifar10_fgsm_adv_pruned_70.pth"),
    "Finetuned_30": Path(
        "checkpoints/resnet18_cifar10_fgsm_adv_pruned_30_finetuned.pth"
    ),
    "Finetuned_50": Path(
        "checkpoints/resnet18_cifar10_fgsm_adv_pruned_50_finetuned.pth"
    ),
    "Finetuned_70": Path(
        "checkpoints/resnet18_cifar10_fgsm_adv_pruned_70_finetuned.pth"
    ),
}


# ---------------- Model Loading ----------------

def load_models():
    loaded_models = {}

    for name, path in MODELS.items():
        print(f"Loading {name}...")

        state = torch.load(
            path,
            map_location=DEVICE,
            weights_only=True
        )

        model = create_resnet18().to(DEVICE)
        model.load_state_dict(state)
        model.eval()

        loaded_models[name] = model

    return loaded_models


# ---------------- Single Measurement ----------------

def benchmark_once(model, batch_size):
    dummy_input = torch.randn(
        batch_size,
        3,
        32,
        32,
        device=DEVICE
    )

    with torch.inference_mode():

        # Warm-up
        for _ in range(WARMUP_STEPS):
            model(dummy_input)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

            baseline_memory = torch.cuda.memory_allocated(DEVICE)
            torch.cuda.reset_peak_memory_stats(DEVICE)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            for _ in range(TIMED_STEPS):
                model(dummy_input)

            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)

            peak_memory_mb = (
                torch.cuda.max_memory_allocated(DEVICE)
                - baseline_memory
            ) / (1024 ** 2)

        else:
            start = time.perf_counter()

            for _ in range(TIMED_STEPS):
                model(dummy_input)

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            peak_memory_mb = None

    return elapsed_ms / TIMED_STEPS, peak_memory_mb


# ---------------- Main ----------------

def main():

    missing = [
        str(path)
        for path in MODELS.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing checkpoints:\n" + "\n".join(missing)
        )

    print("=" * 65)
    print("ResNet-18 Efficiency Benchmark V2")
    print("=" * 65)
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    print(f"Models: {len(MODELS)}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Rounds: {ROUNDS}")
    print(f"Warm-up: {WARMUP_STEPS}")
    print(f"Timed iterations per measurement: {TIMED_STEPS}")

    # Load all models before benchmarking.
    loaded_models = load_models()

    # Create all 24 model/batch-size combinations.
    tasks = [
        (model_name, batch_size)
        for model_name in MODELS
        for batch_size in BATCH_SIZES
    ]

    rng = random.Random(SEED)

    latency_results = {task: [] for task in tasks}
    memory_results = {task: [] for task in tasks}

    # Randomize all combinations independently in each round.
    for round_number in range(ROUNDS):

        round_tasks = tasks.copy()
        rng.shuffle(round_tasks)

        print(f"\nRound {round_number + 1}/{ROUNDS}")

        for model_name, batch_size in round_tasks:

            print(
                f"  {model_name}, batch size {batch_size}"
            )

            latency, peak_memory = benchmark_once(
                loaded_models[model_name],
                batch_size
            )

            task = (model_name, batch_size)

            latency_results[task].append(latency)

            if peak_memory is not None:
                memory_results[task].append(peak_memory)

    # Summarize results.
    results = []

    for model_name, checkpoint_path in MODELS.items():

        checkpoint_size_mb = (
            checkpoint_path.stat().st_size / (1024 ** 2)
        )

        for batch_size in BATCH_SIZES:

            task = (model_name, batch_size)
            latencies = latency_results[task]

            median_latency = statistics.median(latencies)
            mean_latency = statistics.mean(latencies)

            peak_memory = None

            if memory_results[task]:
                peak_memory = statistics.median(
                    memory_results[task]
                )

            results.append({
                "model": model_name,
                "batch_size": batch_size,
                "device": str(DEVICE),
                "checkpoint_size_mb": checkpoint_size_mb,
                "rounds": ROUNDS,
                "measurements": len(latencies),
                "warmup_steps": WARMUP_STEPS,
                "timed_steps_per_measurement": TIMED_STEPS,
                "latency_median_ms": median_latency,
                "latency_mean_ms": mean_latency,
                "latency_stdev_ms": (
                    statistics.stdev(latencies)
                    if len(latencies) > 1 else 0
                ),
                "latency_per_image_ms": (
                    median_latency / batch_size
                ),
                "throughput_images_per_sec": (
                    batch_size * 1000 / median_latency
                ),
                "peak_gpu_memory_mb": peak_memory,
            })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print("Benchmark V2 completed.")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("Previous benchmark files and checkpoints were not modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()
