"""
Compute-load benchmark for the ReID embedding-size variants (256 / 1024 / 2048).

For each model this:
  1. Builds a TensorRT FP16 engine once with a dynamic batch profile
     (min=1, opt/max=MAX_BATCH), via trtexec.
  2. Re-profiles that engine at a range of fixed batch sizes (trtexec
     --loadEngine + --shapes), which is fast since it skips rebuilding.
  3. Parses trtexec's summary GPU Compute latency and throughput.

This isolates the ReID network's own GPU compute cost from the rest of
the DeepStream pipeline (decode, tracker, IO), which is what "computation
load" means when comparing embedding sizes.

Usage:
    python3 benchmark_reid_variants.py [--batch-sizes 1,8,16,32,50] [--skip-build]
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
MAX_BATCH = 50
INPUT_SHAPE = "3x256x128"

MODELS = [
    {
        "name": "256d (production: resnet50_market1501_aicity156)",
        "onnx": "/opt/nvidia/deepstream/deepstream/samples/models/reid/resnet50_market1501_aicity156.onnx",
        "engine": "/tmp/reid_bench_256_gpu0_fp16.engine",
        "input_name": "input",
        "embed_dim": 256,
    },
    {
        "name": "1024d (synthetic projection head)",
        "onnx": "/home/lakshh/workspace/reid/impl_bot/bot_sort_ds_impl/ds_include/onnx_model/bench/resnet50_market1501_1024_synthetic.onnx",
        "engine": "/tmp/reid_bench_1024_gpu0_fp16.engine",
        "input_name": "batched_inputs",
        "embed_dim": 1024,
    },
    {
        "name": "2048d (full resnet50_market1501, no head)",
        "onnx": "/home/lakshh/workspace/reid/impl_bot/bot_sort_ds_impl/ds_include/onnx_model/bench/resnet50_market1501_2048.onnx",
        "engine": "/tmp/reid_bench_2048_gpu0_fp16.engine",
        "input_name": "batched_inputs",
        "embed_dim": 2048,
    },
]


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build_engine(model):
    shapes_min = f"{model['input_name']}:1x{INPUT_SHAPE}"
    shapes_opt = f"{model['input_name']}:{MAX_BATCH}x{INPUT_SHAPE}"
    shapes_max = f"{model['input_name']}:{MAX_BATCH}x{INPUT_SHAPE}"
    cmd = [
        TRTEXEC,
        f"--onnx={model['onnx']}",
        f"--saveEngine={model['engine']}",
        f"--minShapes={shapes_min}",
        f"--optShapes={shapes_opt}",
        f"--maxShapes={shapes_max}",
        "--fp16",
    ]
    result = run(cmd)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise RuntimeError(f"engine build failed for {model['name']}")
    print(f"  built engine -> {model['engine']}")


LATENCY_RE = re.compile(
    r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms"
)
THROUGHPUT_RE = re.compile(r"Throughput: ([\d.]+) qps")


def profile_batch(model, batch_size):
    shapes = f"{model['input_name']}:{batch_size}x{INPUT_SHAPE}"
    cmd = [
        TRTEXEC,
        f"--loadEngine={model['engine']}",
        f"--shapes={shapes}",
        "--avgRuns=50",
        "--iterations=200",
    ]
    result = run(cmd)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise RuntimeError(f"profiling failed for {model['name']} @ batch {batch_size}")

    out = result.stdout
    lat = LATENCY_RE.search(out)
    thr = THROUGHPUT_RE.search(out)
    if not lat or not thr:
        raise RuntimeError(f"could not parse trtexec output for {model['name']} @ batch {batch_size}")

    mean_ms = float(lat.group(3))
    qps = float(thr.group(1))
    return {
        "batch_size": batch_size,
        "mean_latency_ms": mean_ms,
        "latency_per_item_ms": mean_ms / batch_size,
        "throughput_qps": qps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", default="1,8,16,32,50")
    ap.add_argument("--skip-build", action="store_true", help="reuse previously built engines")
    ap.add_argument("--csv", default="/home/lakshh/workspace/reid/impl_bot/bot_sort_ds_impl/reid_variant_benchmark.csv")
    args = ap.parse_args()
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    rows = []
    for model in MODELS:
        print(f"\n=== {model['name']} ===")
        if not args.skip_build or not Path(model["engine"]).exists():
            build_engine(model)
        for bs in batch_sizes:
            if bs > MAX_BATCH:
                print(f"  skip batch {bs} (> engine max batch {MAX_BATCH})")
                continue
            r = profile_batch(model, bs)
            r["model"] = model["name"]
            r["embed_dim"] = model["embed_dim"]
            rows.append(r)
            print(
                f"  batch={bs:>3}  mean={r['mean_latency_ms']:.3f} ms  "
                f"per-item={r['latency_per_item_ms']:.4f} ms  throughput={r['throughput_qps']:.1f} qps"
            )

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "embed_dim", "batch_size", "mean_latency_ms", "latency_per_item_ms", "throughput_qps"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
