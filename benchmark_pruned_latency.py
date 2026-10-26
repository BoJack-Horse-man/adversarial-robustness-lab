
import csv
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from eval_pgd import SimpleCNN


# Configuration
BATCH_SIZES = [1, 64]
WARMUP_STEPS = 20
TIMED_STEPS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE_PATH = Path("checkpoints/simple_cnn.pth")
PRUNED_PATH = Path("checkpoints/simple_cnn_pruned_50.pth")
OUTPUT_PATH = Path("experiments/pruning_latency_results.csv")


def load_model(path):
    model = SimpleCNN().to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def benchmark_model(model, batch_size):
    # Use one fixed batch so data loading does not affect timing.
    dummy_input = torch.randn(
        batch_size, 3, 32, 32, device=DEVICE
    )

    with torch.inference_mode():
        # Warm up the model and GPU.
        for _ in range(WARMUP_STEPS):
            model(dummy_input)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(DEVICE)

        # CUDA events measure GPU execution time.
        if DEVICE.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(TIMED_STEPS):
                model(dummy_input)
            end_event.record()
            torch.cuda.synchronize()

            total_ms = start_event.elapsed_time(end_event)
        else:
            start = time.perf_counter()
            for _ in range(TIMED_STEPS):
                model(dummy_input)
            total_ms = (time.perf_counter() - start) * 1000

        peak_memory_mb = (
            torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
            if DEVICE.type == "cuda"
            else None
        )

    latency_per_batch_ms = total_ms / TIMED_STEPS
    latency_per_image_ms = latency_per_batch_ms / batch_size
    throughput_images_per_sec = (
        batch_size * TIMED_STEPS / (total_ms / 1000)
    )

    return {
        "latency_per_batch_ms": latency_per_batch_ms,
        "latency_per_image_ms": latency_per_image_ms,
        "throughput_images_per_sec": throughput_images_per_sec,
        "peak_gpu_memory_mb": peak_memory_mb,
    }


def main():
    if not BASELINE_PATH.exists() or not PRUNED_PATH.exists():
        raise FileNotFoundError(
            "A checkpoint is missing. Check the paths in this script."
        )

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    models = {
        "baseline": load_model(BASELINE_PATH),
        "pruned_50_percent": load_model(PRUNED_PATH),
    }

    results = []

    for batch_size in BATCH_SIZES:
        for model_name, model in models.items():
            # Repeat once to reduce first-run effects.
            print(
                f"Benchmarking {model_name}, "
                f"batch size {batch_size}..."
            )

            metrics = benchmark_model(model, batch_size)

            row = {
                "model": model_name,
                "batch_size": batch_size,
                "device": str(DEVICE),
                "warmup_steps": WARMUP_STEPS,
                "timed_steps": TIMED_STEPS,
                **metrics,
            }
            results.append(row)

            print(
                f"  Latency: "
                f"{metrics['latency_per_batch_ms']:.4f} ms/batch"
            )
            print(
                f"  Throughput: "
                f"{metrics['throughput_images_per_sec']:.2f} images/s"
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("Checkpoint files were not modified.")


if __name__ == "__main__":
    main()
