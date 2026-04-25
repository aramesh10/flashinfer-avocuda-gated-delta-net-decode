"""
Fast Modal profiler for the custom Triton GDN decode implementation.

Default use:
    python -m modal run scripts/profile_modal.py

Compare against FlashInfer baseline:
    python -m modal run scripts/profile_modal.py --compare

This is intentionally much faster than scripts/run_modal.py:
  - profiles one entry point by default: kernel_fla_recurrent
  - samples one workload per batch size by default
  - uses fewer benchmark iterations/trials
  - writes clean JSON and Markdown summaries to results/
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TRITON_SOURCE_DIR = PROJECT_ROOT / "solution" / "triton"
TRACE_SET_PATH = "/data"
DEFINITION = "gdn_decode_qk4_v8_d128_k_last"

app = modal.App("flashinfer-gdn-profile")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


def read_triton_sources() -> dict[str, str]:
    return {
        str(path.relative_to(TRITON_SOURCE_DIR)): path.read_text()
        for path in TRITON_SOURCE_DIR.rglob("*")
        if path.is_file()
    }


def summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    traces = results["workloads"]
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

    by_batch: dict[str, list[dict[str, Any]]] = {}
    status_counts: dict[str, int] = {}
    for result in traces.values():
        status = result.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        batch_size = str(result.get("axes", {}).get("batch_size", "?"))
        by_batch.setdefault(batch_size, []).append(result)

    batch_summary = {}
    for batch_size, batch_results in by_batch.items():
        batch_latencies = [
            result["latency_ms"]
            for result in batch_results
            if result.get("latency_ms") is not None
        ]
        batch_summary[batch_size] = {
            "count": len(batch_results),
            "mean_latency_ms": statistics.mean(batch_latencies)
            if batch_latencies
            else None,
            "median_latency_ms": statistics.median(batch_latencies)
            if batch_latencies
            else None,
            "min_latency_ms": min(batch_latencies) if batch_latencies else None,
            "max_latency_ms": max(batch_latencies) if batch_latencies else None,
        }

    summary = {
        "entry_point": results["entry_point"],
        "definition": results["definition"],
        "hardware": results["hardware"],
        "benchmark_config": results["benchmark_config"],
        "workload_count": len(traces),
        "status_counts": status_counts,
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "speedup": {
            "mean": statistics.mean(speedups) if speedups else None,
            "median": statistics.median(speedups) if speedups else None,
            "min": min(speedups) if speedups else None,
            "max": max(speedups) if speedups else None,
        },
        "by_batch_size": dict(
            sorted(
                batch_summary.items(),
                key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
            )
        ),
    }
    return summary


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def fmt_speedup(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def get_axes(workload_or_trace: Any) -> dict[str, Any]:
    """FlashInfer-Bench may hand us either a Workload or a Trace wrapper."""
    axes = getattr(workload_or_trace, "axes", None)
    if axes is not None:
        return dict(axes or {})
    workload = getattr(workload_or_trace, "workload", None)
    axes = getattr(workload, "axes", None)
    return dict(axes or {})


def print_summary(summary: dict[str, Any], output_json: Path, output_md: Path) -> None:
    latency = summary["latency_ms"]
    speedup = summary["speedup"]
    statuses = ", ".join(
        f"{status}={count}" for status, count in sorted(summary["status_counts"].items())
    )

    print("\n=== Modal B200 Triton Profile ===")
    print(f"entry point : {summary['entry_point']}")
    print(f"definition  : {summary['definition']}")
    print(f"workloads   : {summary['workload_count']} ({statuses})")
    print(
        "latency ms  : "
        f"mean={fmt_ms(latency['mean'])}, "
        f"median={fmt_ms(latency['median'])}, "
        f"min={fmt_ms(latency['min'])}, "
        f"max={fmt_ms(latency['max'])}"
    )
    print(
        "speedup     : "
        f"mean={fmt_speedup(speedup['mean'])}, "
        f"median={fmt_speedup(speedup['median'])}"
    )
    print("\nBy batch size:")
    print("  B      n    mean ms   median ms   min ms    max ms")
    for batch_size, row in summary["by_batch_size"].items():
        print(
            f"  {batch_size:>3}  "
            f"{row['count']:>3}  "
            f"{fmt_ms(row['mean_latency_ms']):>8}  "
            f"{fmt_ms(row['median_latency_ms']):>9}  "
            f"{fmt_ms(row['min_latency_ms']):>7}  "
            f"{fmt_ms(row['max_latency_ms']):>7}"
        )
    print(f"\nSaved JSON: {output_json}")
    print(f"Saved MD  : {output_md}")


def write_outputs(results: dict[str, Any], summary: dict[str, Any]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"modal_b200_profile_{results['entry_point']}_{stamp}"
    output_json = RESULTS_DIR / f"{stem}.json"
    output_md = RESULTS_DIR / f"{stem}.md"

    payload = {
        "summary": summary,
        "workloads": results["workloads"],
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Modal B200 Triton Profile",
        "",
        f"Entry point: `{summary['entry_point']}`",
        "",
        f"Definition: `{summary['definition']}`",
        "",
        f"Hardware: {summary['hardware']}",
        "",
        "Benchmark config:",
        "",
        f"- `warmup_runs = {summary['benchmark_config']['warmup_runs']}`",
        f"- `iterations = {summary['benchmark_config']['iterations']}`",
        f"- `num_trials = {summary['benchmark_config']['num_trials']}`",
        f"- `one_per_batch = {summary['benchmark_config']['one_per_batch']}`",
        "",
        "## Overall",
        "",
        "| Workloads | Mean Latency (ms) | Median Latency (ms) | Min (ms) | Max (ms) | Mean Speedup | Median Speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {summary['workload_count']} "
            f"| {fmt_ms(summary['latency_ms']['mean'])} "
            f"| {fmt_ms(summary['latency_ms']['median'])} "
            f"| {fmt_ms(summary['latency_ms']['min'])} "
            f"| {fmt_ms(summary['latency_ms']['max'])} "
            f"| {fmt_speedup(summary['speedup']['mean'])} "
            f"| {fmt_speedup(summary['speedup']['median'])} |"
        ),
        "",
        "## By Batch Size",
        "",
        "| Batch Size | Workloads | Mean Latency (ms) | Median Latency (ms) | Min (ms) | Max (ms) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for batch_size, row in summary["by_batch_size"].items():
        lines.append(
            f"| {batch_size} "
            f"| {row['count']} "
            f"| {fmt_ms(row['mean_latency_ms'])} "
            f"| {fmt_ms(row['median_latency_ms'])} "
            f"| {fmt_ms(row['min_latency_ms'])} "
            f"| {fmt_ms(row['max_latency_ms'])} |"
        )
    output_md.write_text("\n".join(lines) + "\n")
    return output_json, output_md


@app.function(image=image, gpu="B200", timeout=1800, volumes={TRACE_SET_PATH: trace_volume})
def run_profile(
    entry_point: str,
    sources: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    benchmark_config = BenchmarkConfig(
        warmup_runs=config["warmup_runs"],
        iterations=config["iterations"],
        num_trials=config["num_trials"],
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
            definition=DEFINITION,
            author="team-name",
        )


    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[DEFINITION]
    workloads = trace_set.workloads.get(DEFINITION, [])

    if config["one_per_batch"]:
        selected = {}
        for workload in workloads:
            batch_size = get_axes(workload).get("batch_size", "?")
            selected.setdefault(batch_size, workload)
        workloads = [
            selected[key]
            for key in sorted(
                selected,
                key=lambda item: int(item) if str(item).isdigit() else str(item),
            )
        ]

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, benchmark_config)
    result_trace_set = benchmark.run_all(dump_traces=False)

    profile_results: dict[str, Any] = {
        "entry_point": entry_point,
        "definition": DEFINITION,
        "hardware": "Modal B200",
        "benchmark_config": config,
        "workloads": {},
    }
    for trace in result_trace_set.traces.get(DEFINITION, []):
        if not trace.evaluation:
            continue
        entry = {
            "status": trace.evaluation.status.value,
            "axes": dict(trace.workload.axes or {}),
        }
        if trace.evaluation.performance:
            entry["latency_ms"] = trace.evaluation.performance.latency_ms
            entry["reference_latency_ms"] = (
                trace.evaluation.performance.reference_latency_ms
            )
            entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
        if trace.evaluation.correctness:
            entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
            entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
        profile_results["workloads"][trace.workload.uuid] = entry

    return profile_results


def print_comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    """Print a side-by-side table of baseline vs candidate latency by batch size."""
    b_by_batch = baseline["by_batch_size"]
    c_by_batch = candidate["by_batch_size"]
    all_batches = sorted(
        set(b_by_batch) | set(c_by_batch),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    baseline_ep = baseline["entry_point"]
    candidate_ep = candidate["entry_point"]

    print(f"\n=== Comparison: {candidate_ep} vs {baseline_ep} (FlashInfer baseline) ===")
    print(f"{'B':>4}  {'baseline ms':>12}  {'ours ms':>10}  {'speedup':>9}")
    print("-" * 44)
    speedups = []
    for b in all_batches:
        b_lat = (b_by_batch.get(b) or {}).get("median_latency_ms")
        c_lat = (c_by_batch.get(b) or {}).get("median_latency_ms")
        if b_lat and c_lat:
            sp = b_lat / c_lat
            speedups.append(sp)
            print(f"{b:>4}  {fmt_ms(b_lat):>12}  {fmt_ms(c_lat):>10}  {fmt_speedup(sp):>9}")
        else:
            print(f"{b:>4}  {fmt_ms(b_lat):>12}  {fmt_ms(c_lat):>10}  {'n/a':>9}")
    if speedups:
        print("-" * 44)
        print(f"{'mean':>4}  {'':>12}  {'':>10}  {fmt_speedup(statistics.mean(speedups)):>9}")
        print(f"{'median':>6}  {'':>10}  {'':>10}  {fmt_speedup(statistics.median(speedups)):>9}")


@app.local_entrypoint()
def main(
    entry_point: str = "kernel_fla_recurrent",
    warmup_runs: int = 1,
    iterations: int = 20,
    num_trials: int = 1,
    one_per_batch: bool = True,
    compare: bool = False,
):
    if not TRITON_SOURCE_DIR.exists():
        raise FileNotFoundError(f"Missing Triton source directory: {TRITON_SOURCE_DIR}")

    config = {
        "warmup_runs": warmup_runs,
        "iterations": iterations,
        "num_trials": num_trials,
        "one_per_batch": one_per_batch,
    }

    print("Reading Triton sources...")
    sources = read_triton_sources()

    if compare:
        print(f"Profiling kernel_fi_baseline and {entry_point} on Modal B200...")
        # Run both in parallel
        baseline_fut = run_profile.spawn("kernel_fi_baseline", sources, config)
        candidate_fut = run_profile.spawn(entry_point, sources, config)
        baseline_results = baseline_fut.get()
        candidate_results = candidate_fut.get()

        baseline_summary = summarize_results(baseline_results)
        candidate_summary = summarize_results(candidate_results)

        b_json, b_md = write_outputs(baseline_results, baseline_summary)
        c_json, c_md = write_outputs(candidate_results, candidate_summary)

        print_summary(baseline_summary, b_json, b_md)
        print_summary(candidate_summary, c_json, c_md)
        print_comparison(baseline_summary, candidate_summary)
    else:
        print(f"Profiling {entry_point} on Modal B200...")
        print(
            "Config: "
            f"warmup={warmup_runs}, iterations={iterations}, "
            f"trials={num_trials}, one_per_batch={one_per_batch}"
        )
        results = run_profile.remote(entry_point, sources, config)
        summary = summarize_results(results)
        output_json, output_md = write_outputs(results, summary)
        print_summary(summary, output_json, output_md)


if __name__ == "__main__":
    sys.exit("Run with: python -m modal run profile_modal.py")
