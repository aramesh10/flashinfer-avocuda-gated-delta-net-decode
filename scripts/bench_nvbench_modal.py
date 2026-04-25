"""
Run the NVBench GDN decode benchmark on Modal B200.

Default:
    python -m modal run scripts/bench_nvbench_modal.py

Examples:
    python -m modal run scripts/bench_nvbench_modal.py --algo=fla-recurrent --batch-size=1
    python -m modal run scripts/bench_nvbench_modal.py --algo=all --batch-size=64 --nvbench-args="--min-time 0.5"
"""

from __future__ import annotations

import json
import shlex
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TRACE_SET_PATH = "/data"

app = modal.App("flashinfer-gdn-nvbench")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "cuda-bench",
        "flashinfer-bench",
        "safetensors",
        "torch",
        "triton",
        "numpy",
    )
)


def read_sources() -> dict[str, str]:
    roots = [
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "solution" / "triton",
    ]
    sources = {}
    for root in roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            relative_path = path.relative_to(PROJECT_ROOT).as_posix()
            sources[relative_path] = path.read_text()
    return sources


@app.function(image=image, gpu="B200", timeout=1800, volumes={TRACE_SET_PATH: trace_volume})
def run_nvbench_remote(
    sources: dict[str, str],
    algo: str,
    batch_size: int | None,
    workload_index: int,
    nvbench_args: list[str],
) -> dict[str, Any]:
    import os
    import subprocess

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        for relative_path, content in sources.items():
            output_path = root / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)

        (root / "scripts").mkdir(parents=True, exist_ok=True)
        (root / "solution" / "triton").mkdir(parents=True, exist_ok=True)
        (root / "scripts" / "__init__.py").write_text("")
        (root / "solution" / "__init__.py").write_text("")
        (root / "solution" / "triton" / "__init__.py").write_text("")
        expected = root / "scripts" / "bench_nvbench.py"
        if not expected.exists():
            raise FileNotFoundError(
                f"Missing staged NVBench script: {expected}. "
                f"Staged files: {sorted(sources)}"
            )

        env = dict(os.environ)
        env["FIB_DATASET_PATH"] = TRACE_SET_PATH
        env["PYTHONPATH"] = str(root)

        command = [
            sys.executable,
            "-m",
            "scripts.bench_nvbench",
            f"--algo={algo}",
            f"--workload-index={workload_index}",
        ]
        if batch_size is not None:
            command.append(f"--batch-size={batch_size}")
        if nvbench_args:
            command.extend(["--", *nvbench_args])

        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        result_files = {}
        remote_results_dir = root / "results"
        if remote_results_dir.exists():
            for path in remote_results_dir.iterdir():
                if path.is_file():
                    result_files[path.name] = path.read_text(errors="replace")

        return {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
            "command": command,
            "result_files": result_files,
            "algo": algo,
            "batch_size": batch_size,
            "workload_index": workload_index,
            "nvbench_args": nvbench_args,
        }


def write_outputs(payload: dict[str, Any]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"modal_nvbench_gdn_decode_{payload['algo']}_{stamp}"
    stdout_path = RESULTS_DIR / f"{stem}.txt"
    manifest_path = RESULTS_DIR / f"{stem}.json"

    stdout_parts = [payload["stdout"]]
    for name, content in sorted(payload["result_files"].items()):
        if name.endswith(".txt") and content not in stdout_parts:
            stdout_parts.append(f"\n\n# Remote file: {name}\n\n{content}")
    stdout_path.write_text("\n".join(stdout_parts))

    manifest = {
        "algo": payload["algo"],
        "batch_size": payload["batch_size"],
        "command": payload["command"],
        "returncode": payload["returncode"],
        "workload_index": payload["workload_index"],
        "nvbench_args": payload["nvbench_args"],
        "stdout_path": str(stdout_path),
        "remote_result_files": sorted(payload["result_files"]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return stdout_path, manifest_path


@app.local_entrypoint()
def main(
    algo: str = "fla-recurrent",
    batch_size: int | None = 1,
    workload_index: int = 0,
    nvbench_args: str = "",
):
    sources = read_sources()
    parsed_nvbench_args = shlex.split(nvbench_args)

    print("Running NVBench on Modal B200...")
    print(f"algo           : {algo}")
    print(f"batch_size     : {batch_size if batch_size is not None else 'first'}")
    print(f"workload_index : {workload_index}")
    if parsed_nvbench_args:
        print(f"nvbench_args   : {' '.join(parsed_nvbench_args)}")

    payload = run_nvbench_remote.remote(
        sources,
        algo,
        batch_size,
        workload_index,
        parsed_nvbench_args,
    )
    stdout_path, manifest_path = write_outputs(payload)

    print("\n=== NVBench Output ===")
    print(payload["stdout"])
    if payload.get("stderr"):
        print("\n=== NVBench Stderr ===")
        print(payload["stderr"])
    if payload["returncode"] != 0:
        raise RuntimeError(f"NVBench subprocess failed with code {payload['returncode']}")
    print(f"Saved stdout  : {stdout_path}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    sys.exit("Run with: python -m modal run scripts/bench_nvbench_modal.py")
