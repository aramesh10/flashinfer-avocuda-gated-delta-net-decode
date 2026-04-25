"""
Measure pure kernel execution time on Modal B200 using CUDA event timing.

CUDA events are placed immediately before and after the kernel dispatch —
no Python interpreter overhead, no benchmark framework, no L2 flush between
iterations (so state fits in L2 cache, which is realistic for decode).

Usage:
    modal run scripts/time_kernel_modal.py
    modal run scripts/time_kernel_modal.py --algo fla-tma
    modal run scripts/time_kernel_modal.py --warmup 200 --iters 1000

Output: mean/min/median kernel time in µs per batch size.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRITON_SOURCE_DIR = PROJECT_ROOT / "solution" / "triton"
TRACE_SET_PATH = "/data"

app = modal.App("flashinfer-gdn-time-kernel")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "safetensors", "torch", "triton", "numpy")
)


def read_triton_sources() -> dict[str, str]:
    return {
        str(path.relative_to(TRITON_SOURCE_DIR)): path.read_text()
        for path in TRITON_SOURCE_DIR.rglob("*")
        if path.is_file()
    }


@app.function(image=image, gpu="B200+", timeout=1800, volumes={TRACE_SET_PATH: trace_volume})
def time_kernel(
    sources: dict[str, str],
    algo: str,
    batch_sizes: list[int],
    warmup: int,
    iters: int,
) -> list[dict]:
    import importlib
    import os
    import sys
    import tempfile
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    # Stage source files — keep tmp_dir alive through import
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    source_dir = Path(tmp_dir) / "solution" / "triton"
    source_dir.mkdir(parents=True)
    for relative_path, content in sources.items():
        out = source_dir / relative_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content)

    sys.path.insert(0, str(Path(tmp_dir)))
    (Path(tmp_dir) / "solution" / "__init__.py").write_text("")
    (Path(tmp_dir) / "solution" / "triton" / "__init__.py").write_text("")

    # Clear any stale cached modules so re-imports hit the staged files
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("solution"):
            del sys.modules[mod_name]

    print(f"[stage] staged files: {sorted(sources.keys())}")
    print(f"[stage] sys.path[0]: {sys.path[0]}")

    ALGO_MAP = {
        "fla-recurrent":   ("solution.triton.kernel",   "kernel_fla_recurrent"),
        "fla-tma":         ("solution.triton.kernel",   "kernel_fla_tma"),
        "fi-baseline":     ("solution.triton.kernel",   "kernel_fi_baseline"),
    }
    module_name, fn_name = ALGO_MAP[algo]
    mod = importlib.import_module(module_name)
    kernel_fn = getattr(mod, fn_name)

    # Load one workload per batch size from the trace set
    import json
    workload_path = Path(TRACE_SET_PATH) / "workloads" / "gdn" / "gdn_decode_qk4_v8_d128_k_last.jsonl"
    workloads_by_batch: dict[int, dict] = {}
    with workload_path.open() as f:
        for line in f:
            w = json.loads(line)
            w = w.get("workload", w)
            b = w.get("axes", {}).get("batch_size")
            if b is not None and b not in workloads_by_batch:
                workloads_by_batch[b] = w

    def load_tensors(workload: dict) -> dict:
        tensors = {}
        dataset_path = Path(TRACE_SET_PATH)
        for name, spec in workload["inputs"].items():
            if spec["type"] == "scalar":
                tensors[name] = spec["value"]
            else:
                path = (dataset_path / spec["path"]).resolve()
                tensors[name] = load_file(str(path), device="cuda")[spec["tensor_key"]]
        tensors["output"] = torch.empty_like(tensors["v"])
        tensors["new_state"] = torch.empty_like(tensors["state"])
        return tensors

    results = []
    for B in batch_sizes:
        if B not in workloads_by_batch:
            print(f"  B={B}: no workload found, skipping")
            continue

        tensors = load_tensors(workloads_by_batch[B])

        # Warmup: trigger JIT compilation + cache warm
        for _ in range(warmup):
            kernel_fn(**tensors)
        torch.cuda.synchronize()

        # Measure: CUDA events placed around kernel dispatch only
        # Re-use same tensors every iteration (state changes but that's fine —
        # we're measuring dispatch time, not correctness).
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_us = []

        for _ in range(iters):
            start.record()
            kernel_fn(**tensors)
            end.record()
            torch.cuda.synchronize()
            times_us.append(start.elapsed_time(end) * 1000)  # ms → µs

        mean_us  = statistics.mean(times_us)
        min_us   = min(times_us)
        med_us   = statistics.median(times_us)
        p99_us   = sorted(times_us)[int(0.99 * len(times_us))]

        # Compute roofline for this B
        HV, V, K = 8, 128, 128
        bw_tbs    = 8.0      # B200 HBM TB/s
        fp32_tflops = 67.0   # B200 f32 TFLOPS
        mem_bytes = 8 * B * HV * V * K          # state read+write (f32)
        flops     = 7 * B * HV * V * K          # from roofline equation
        t_mem_us  = (mem_bytes / (bw_tbs * 1e12)) * 1e6
        t_cmp_us  = (flops    / (fp32_tflops * 1e12)) * 1e6
        sol_us    = max(t_mem_us, t_cmp_us)

        results.append({
            "batch_size": B,
            "mean_us":    round(mean_us, 4),
            "min_us":     round(min_us,  4),
            "median_us":  round(med_us,  4),
            "p99_us":     round(p99_us,  4),
            "sol_mem_us": round(t_mem_us, 4),
            "sol_cmp_us": round(t_cmp_us, 4),
            "sol_us":     round(sol_us,   4),
            "efficiency": round(sol_us / mean_us * 100, 2),
        })

    return results


def print_single(results: list[dict], algo: str) -> None:
    print(f"\n{'B':>4}  {'mean µs':>9}  {'min µs':>8}  {'median µs':>10}  {'p99 µs':>8}  {'SoL µs':>8}  {'efficiency':>10}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['batch_size']:>4}  "
            f"{r['mean_us']:>9.2f}  "
            f"{r['min_us']:>8.2f}  "
            f"{r['median_us']:>10.2f}  "
            f"{r['p99_us']:>8.2f}  "
            f"{r['sol_us']:>8.4f}  "
            f"{r['efficiency']:>9.2f}%"
        )
    print()
    print("SoL = max(8·N·HV·V·K / BW, 7·N·HV·V·K / FLOPS)")
    print("BW = 8 TB/s (B200 HBM),  FLOPS = 67 TFLOPS f32")


def print_comparison(baseline: list[dict], candidate: list[dict], baseline_algo: str, candidate_algo: str) -> None:
    b_by_batch = {r["batch_size"]: r for r in baseline}
    c_by_batch = {r["batch_size"]: r for r in candidate}
    all_batches = sorted(set(b_by_batch) | set(c_by_batch))

    print(f"\n=== {candidate_algo} vs {baseline_algo} ===")
    print(f"{'B':>4}  {'baseline µs':>12}  {'ours µs':>9}  {'speedup':>9}  {'SoL µs':>8}  {'SoL eff':>8}")
    print("-" * 60)
    speedups = []
    for B in all_batches:
        b = b_by_batch.get(B)
        c = c_by_batch.get(B)
        b_t = b["mean_us"] if b else None
        c_t = c["mean_us"] if c else None
        sol = (c or b or {}).get("sol_us", 0)
        eff = (c or b or {}).get("efficiency", 0)
        sp = b_t / c_t if (b_t and c_t) else None
        if sp:
            speedups.append(sp)
        print(
            f"{B:>4}  "
            f"{f'{b_t:.2f}' if b_t else 'n/a':>12}  "
            f"{f'{c_t:.2f}' if c_t else 'n/a':>9}  "
            f"{f'{sp:.3f}x' if sp else 'n/a':>9}  "
            f"{sol:>8.4f}  "
            f"{eff:>7.2f}%"
        )
    if speedups:
        print("-" * 60)
        print(f"{'mean':>4}  {'':>12}  {'':>9}  {statistics.mean(speedups):>8.3f}x")
        print(f"{'median':>6}  {'':>10}  {'':>9}  {statistics.median(speedups):>8.3f}x")


@app.local_entrypoint()
def main(
    algo: str = "fla-recurrent",
    warmup: int = 200,
    iters: int = 500,
    batch_sizes: str = "1,4,8,16,32,48,64",
    compare: bool = False,
    baseline: str = "fi-baseline",
):
    bs = [int(x) for x in batch_sizes.split(",")]
    sources = read_triton_sources()

    if compare:
        print(f"Timing {baseline} vs {algo} on Modal B200 (parallel)")
        print(f"warmup={warmup}, iters={iters}, batch_sizes={bs}\n")
        baseline_fut = time_kernel.spawn(sources, baseline, bs, warmup, iters)
        candidate_fut = time_kernel.spawn(sources, algo, bs, warmup, iters)
        baseline_results = baseline_fut.get()
        candidate_results = candidate_fut.get()
        print(f"\n--- {baseline} ---")
        print_single(baseline_results, baseline)
        print(f"\n--- {algo} ---")
        print_single(candidate_results, algo)
        print_comparison(baseline_results, candidate_results, baseline, algo)
    else:
        print(f"Timing {algo} on Modal B200")
        print(f"warmup={warmup}, iters={iters}, batch_sizes={bs}")
        results = time_kernel.remote(sources, algo, bs, warmup, iters)
        print_single(results, algo)
