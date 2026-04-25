"""
Benchmark GDN decode kernels using NVBench (NVIDIA's official kernel benchmarking tool).

Validates timing against an independent, NVIDIA-standard benchmark. NVBench uses
CUDA events on a dedicated stream with statistical convergence criteria.

Usage:
    python -m scripts.bench_nvbench --algo=fla-recurrent
    python -m scripts.bench_nvbench --algo=fi-baseline
    python -m scripts.bench_nvbench --algo=all

Useful examples:
    python -m scripts.bench_nvbench --algo=fla-recurrent --batch-size=1
    python -m scripts.bench_nvbench --algo=all --batch-size=64 -- --min-time 0.5
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cuda.bench as bench
import torch

from .profile_proton import load_workload_tensors
from .shared import DEFS, load_algo_functions, resolve_algo_names



PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

ALGOS = load_algo_functions()
RUNTIME_CONFIG: dict[str, Any] = {
    "batch_size": None,
    "workload_index": 0,
}


def as_torch_stream(cs: bench.CudaStream) -> torch.cuda.ExternalStream:
    return torch.cuda.ExternalStream(cs.addressof())


def gdn_decode(state: bench.State):
    """Benchmark GDN decode kernels."""
    algo_name = state.get_string("Algo")
    kernel_fn = ALGOS[algo_name]

    tensors = load_workload_tensors(
        DEFS.DECODE,
        batch_size=RUNTIME_CONFIG["batch_size"],
        workload_index=RUNTIME_CONFIG["workload_index"],
    )

    h_state = tensors["state"]
    state_bytes = h_state.nelement() * h_state.element_size()
    qkv_bytes = sum(
        tensors[k].nelement() * tensors[k].element_size() for k in ("q", "k", "v")
    )
    output_bytes = tensors["output"].nelement() * tensors["output"].element_size()

    state.add_global_memory_reads(state_bytes + qkv_bytes, column_name="Read")
    state.add_global_memory_writes(state_bytes + output_bytes, column_name="Write")

    def launcher(launch: bench.Launch):
        stream = as_torch_stream(launch.get_stream())
        with torch.cuda.stream(stream):
            kernel_fn(**tensors)

    # batched=False enables L2 cache flushing between iterations.
    state.exec(launcher, batched=False)


def make_result_paths() -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"nvbench_gdn_decode_{stamp}"
    return RESULTS_DIR / f"{stem}.json", RESULTS_DIR / f"{stem}.txt"


def write_manifest(
    *,
    manifest_path: Path,
    stdout_path: Path,
    algo_names: list[str],
    batch_size: int | None,
    workload_index: int,
    nvbench_args: list[str],
) -> None:
    manifest = {
        "definition": DEFS.DECODE,
        "algorithms": algo_names,
        "batch_size": batch_size,
        "workload_index": workload_index,
        "nvbench_args": nvbench_args,
        "stdout_path": str(stdout_path),
        "command": ["python", "-m", "scripts.bench_nvbench", *sys.argv[1:]],
        "note": (
            "NVBench timing/statistical output is saved in stdout_path. This "
            "manifest records the exact benchmark configuration used for the run."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NVBench GDN decode benchmark")
    parser.add_argument(
        "--algo",
        default="all",
        help="Algorithm(s) to benchmark. Comma-separated or 'all' (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Only benchmark workloads with this batch size. Defaults to first workload.",
    )
    parser.add_argument(
        "--workload-index",
        type=int,
        default=0,
        help="Index among matching workloads for the selected batch size.",
    )
    args, remaining = parser.parse_known_args()

    algo_names = resolve_algo_names(args.algo)
    RUNTIME_CONFIG["batch_size"] = args.batch_size
    RUNTIME_CONFIG["workload_index"] = args.workload_index

    manifest_path, stdout_path = make_result_paths()
    write_manifest(
        manifest_path=manifest_path,
        stdout_path=stdout_path,
        algo_names=algo_names,
        batch_size=args.batch_size,
        workload_index=args.workload_index,
        nvbench_args=remaining,
    )

    print("=== NVBench GDN Decode ===")
    print(f"definition     : {DEFS.DECODE}")
    print(f"algorithms     : {', '.join(algo_names)}")
    print(f"batch_size     : {args.batch_size if args.batch_size is not None else 'first'}")
    print(f"workload_index : {args.workload_index}")
    print(f"manifest       : {manifest_path}")
    print(f"nvbench stdout : {stdout_path}")
    print("")

    b = bench.register(gdn_decode)
    b.add_string_axis("Algo", algo_names)
    bench.run_all_benchmarks(["bench_nvbench", *remaining])


if __name__ == "__main__":
    main()
