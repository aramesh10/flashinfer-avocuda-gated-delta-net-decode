"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import statistics
import sys
import tempfile
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(entry_point: str, sources: dict[str, str], config: dict | None = None, filter: dict | None = None) -> dict:
    """Run one benchmark on Modal B200 and return results."""
    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    if config is None:
        config = {}
    benchmark_config = BenchmarkConfig(
        warmup_runs=config.get("warmup_runs", 3),
        iterations=config.get("iterations", 100),
        num_trials=config.get("num_trials", 5),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        source_dir = Path(tmp_dir) / "solution" / "triton"
        source_dir.mkdir(parents=True)
        for relative_path, content in sources.items():
            output_path = source_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)

        spec = BuildSpec(
            language="triton",
            target_hardware=["cuda"],
            entry_point=f"kernel.py::{entry_point}",
            destination_passing_style=True,
        )
        solution = pack_solution_from_files(
            path=str(source_dir),
            spec=spec,
            name=entry_point,
            definition="gdn_decode_qk4_v8_d128_k_last",
            author="team-name",
        )

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    # Optional workload filtering
    if filter:
        batch_sizes = filter.get("batch_sizes")      # list[int] | None
        max_per_batch = filter.get("max_per_batch")  # int | None
        if batch_sizes is not None:
            batch_sizes_set = set(batch_sizes)
            workloads = [w for w in workloads if (dict(w.axes or {}).get("batch_size")) in batch_sizes_set]
        if max_per_batch is not None:
            seen: dict[int, int] = {}
            filtered = []
            for w in workloads:
                b = dict(w.axes or {}).get("batch_size")
                if seen.get(b, 0) < max_per_batch:
                    filtered.append(w)
                    seen[b] = seen.get(b, 0) + 1
            workloads = filtered

    if not workloads:
        raise ValueError(f"No workloads remain after filtering with filter={filter}")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, benchmark_config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            solution_name = trace.solution
            workload = trace.workload
            entry = {
                "status": trace.evaluation.status.value,
                "solution": solution_name,
                "axes": dict(workload.axes or {}),
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name].setdefault(solution_name, {})[workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, solutions in results.items():
        print(f"\n{def_name}:")
        for solution_name, traces in solutions.items():
            status_counts = {}
            for result in traces.values():
                status = result.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            latencies = [
                result["latency_ms"]
                for result in traces.values()
                if result.get("latency_ms") is not None
            ]
            speedups = [
                result["speedup_factor"]
                for result in traces.values()
                if result.get("speedup_factor") is not None
            ]
            statuses = ", ".join(
                f"{status}={count}" for status, count in sorted(status_counts.items())
            )
            print(f"\n  {solution_name}: {len(traces)} workloads ({statuses})")
            if latencies:
                print(
                    "    latency ms: "
                    f"mean={statistics.mean(latencies):.4f}, "
                    f"median={statistics.median(latencies):.4f}, "
                    f"min={min(latencies):.4f}, max={max(latencies):.4f}"
                )
            if speedups:
                print(
                    "    speedup: "
                    f"mean={statistics.mean(speedups):.3f}x, "
                    f"median={statistics.median(speedups):.3f}x"
                )

            by_batch = {}
            for workload_uuid, result in traces.items():
                axes = result.get("axes", {})
                batch_size = axes.get("batch_size", "?")
                by_batch.setdefault(batch_size, []).append((workload_uuid, result))

            for batch_size in sorted(by_batch, key=lambda item: int(item) if str(item).isdigit() else str(item)):
                batch_results = by_batch[batch_size]
                batch_latencies = [
                    result["latency_ms"]
                    for _, result in batch_results
                    if result.get("latency_ms") is not None
                ]
                if batch_latencies:
                    print(
                        f"    B={batch_size}: "
                        f"n={len(batch_results)}, "
                        f"mean={statistics.mean(batch_latencies):.4f} ms, "
                        f"median={statistics.median(batch_latencies):.4f} ms"
                    )
                else:
                    statuses = sorted({result.get("status") for _, result in batch_results})
                    print(f"    B={batch_size}: n={len(batch_results)}, statuses={statuses}")


def read_triton_sources() -> dict[str, str]:
    """Read source files so they can be sent to Modal without local bench deps."""
    source_dir = PROJECT_ROOT / "solution" / "triton"
    return {
        str(path.relative_to(source_dir)): path.read_text()
        for path in source_dir.rglob("*")
        if path.is_file()
    }


@app.local_entrypoint()
def main(
    variants: str = "kernel_fi_baseline,kernel_fla_recurrent,kernel_fla_tma",
    batch_sizes: str = "",
    max_per_batch: int = 0,
):
    """
    Run benchmarks on Modal B200.

    Examples:
        # All variants, all workloads (default)
        modal run scripts/run_modal.py

        # One variant only
        modal run scripts/run_modal.py --variants kernel_fla_recurrent

        # Only batch sizes 1 and 8
        modal run scripts/run_modal.py --batch-sizes 1,8

        # One workload per batch size (fast sanity check)
        modal run scripts/run_modal.py --max-per-batch 1

        # Combined
        modal run scripts/run_modal.py --variants kernel_fla_recurrent --batch-sizes 1,64 --max-per-batch 1
    """
    variant_list = [v.strip() for v in variants.split(",") if v.strip()]

    filter: dict | None = None
    if batch_sizes or max_per_batch:
        filter = {}
        if batch_sizes:
            filter["batch_sizes"] = [int(b) for b in batch_sizes.split(",") if b.strip()]
        if max_per_batch:
            filter["max_per_batch"] = max_per_batch

    print("Reading Triton source files...")
    sources = read_triton_sources()
    if filter:
        print(f"Filter: {filter}")

    for variant in variant_list:
        print(f"\nRunning benchmark on Modal B200 for {variant}...")
        results = run_benchmark.remote(variant, sources, filter=filter)

        if not results:
            print("No results returned!")
            continue

        print_results(results)
