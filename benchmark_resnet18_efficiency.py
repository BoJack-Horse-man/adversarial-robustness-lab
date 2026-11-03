
import csv
import time
from pathlib import Path

import torch

from models.resnet_cifar import create_resnet18


# ---------------- Configuration ----------------

BATCH_SIZES = [1, 32, 64]
WARMUP_STEPS = 20
TIMED_STEPS = 100
TRIALS = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_PATH = Path("experiments/resnet18_efficiency_results.csv")

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

def load_model(path):
    model = create_resnet18().to(DEVICE)

    state = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    return model


# ---------------- Benchmark ----------------

def benchmark_model(model, batch_size):
    torch.manual_seed(SEED)

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

        trial_latencies = []

        for trial in range(TRIALS):

            if DEVICE.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()

                for _ in range(TIMED_STEPS):
                    model(dummy_input)

                end_event.record()
                torch.cuda.synchronize()

                elapsed_ms = start_event.elapsed_time(end_event)

            else:
                start = time.perf_counter()

                for _ in range(TIMED_STEPS):
                    model(dummy_input)

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

            trial_latencies.append(elapsed_ms / TIMED_STEPS)

        peak_memory_mb = (
        (torch.cuda.max_memory_allocated(DEVICE) - baseline_memory)
        / (1024 ** 2)
        if DEVICE.type == "cuda"
        else None
    )

    # Use the median across trials.
    trial_latencies.sort()
    median_batch_latency = trial_latencies[len(trial_latencies) // 2]

    return {
        "latency_per_batch_ms": median_batch_latency,
        "latency_per_image_ms": median_batch_latency / batch_size,
        "throughput_images_per_sec": (
            batch_size * 1000 / median_batch_latency
        ),
        "peak_gpu_memory_mb": peak_memory_mb,
    }


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

    print("=" * 60)
    print("ResNet-18 Inference Efficiency Benchmark")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    print(f"Models: {len(MODELS)}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Trials: {TRIALS}")
    print(f"Timed iterations per trial: {TIMED_STEPS}")

    results = []

    for model_name, checkpoint_path in MODELS.items():

        print(f"\nLoading {model_name}...")

        model = load_model(checkpoint_path)

        checkpoint_size_mb = (
            checkpoint_path.stat().st_size / (1024 ** 2)
        )

        for batch_size in BATCH_SIZES:

            print(
                f"Benchmarking {model_name}, "
                f"batch size {batch_size}..."
            )

            metrics = benchmark_model(model, batch_size)

            row = {
                "model": model_name,
                "batch_size": batch_size,
                "device": str(DEVICE),
                "checkpoint_size_mb": checkpoint_size_mb,
                "warmup_steps": WARMUP_STEPS,
                "timed_steps_per_trial": TIMED_STEPS,
                "trials": TRIALS,
                **metrics,
            }

            results.append(row)

            print(
                f"  Latency: "
                f"{metrics['latency_per_batch_ms']:.4f} ms/batch"
            )

            print(
                f"  Throughput: "
                f"{metrics['throughput_images_per_sec']:.2f} images/sec"
            )

            if metrics["peak_gpu_memory_mb"] is not None:
                print(
                    f"  Peak GPU memory: "
                    f"{metrics['peak_gpu_memory_mb']:.2f} MB"
                )

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

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

    print("\n" + "=" * 60)
    print(f"Benchmark completed.")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("Original checkpoints were not modified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
