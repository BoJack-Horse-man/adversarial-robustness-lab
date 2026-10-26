# 50% Pruning: Inference Latency Benchmark

**Date:** 2026-10-26
**Device:** NVIDIA GeForce GTX 1050, CUDA
**Method:** 20 warm-up passes and 100 timed forward passes per model and batch size, using fixed GPU-resident inputs and inference mode.

## Results

| Batch size | Model | Latency per batch (ms) | Throughput (images/s) |
|---:|---|---:|---:|
| 1 | Baseline | 0.8415 | 1188.39 |
| 1 | 50% pruned | 0.8272 | 1208.96 |
| 64 | Baseline | 5.9306 | 10791.47 |
| 64 | 50% pruned | 6.2054 | 10313.65 |

## Interpretation

The pruned model was approximately 1.7% faster at batch size 1 but approximately 4.6% slower at batch size 64. These results do not demonstrate a consistent inference-speed improvement from unstructured magnitude pruning.

## Limitations

This was one benchmark run. Timing variation, GPU activity, and measurement noise may affect the small differences. Repeated measurements are needed before drawing firm conclusions. Fixed inputs were already on the GPU, so data-loading and host-to-device transfer time were not measured.

**Raw results:** `experiments/pruning_latency_results.csv`
