"""Tensor-loading helpers shared by profiling scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "mlsys26-contest"


def get_dataset_path() -> Path:
    return Path(os.environ.get("FIB_DATASET_PATH", DEFAULT_DATASET_PATH)).resolve()


def _trace_workload(trace: dict[str, Any]) -> dict[str, Any]:
    workload = trace.get("workload", trace)
    if "workload" in workload:
        workload = workload["workload"]
    return workload


def load_workload_record(
    definition: str,
    *,
    batch_size: int | None = None,
    workload_index: int = 0,
) -> dict[str, Any]:
    """Load one workload record from the FlashInfer Trace dataset."""
    dataset_path = get_dataset_path()
    workload_path = dataset_path / "workloads" / "gdn" / f"{definition}.jsonl"
    if not workload_path.exists():
        raise FileNotFoundError(f"Workload file not found: {workload_path}")

    matches = []
    with workload_path.open("r", encoding="utf-8") as f:
        for line in f:
            trace = json.loads(line)
            workload = _trace_workload(trace)
            axes = workload.get("axes", {})
            if batch_size is None or axes.get("batch_size") == batch_size:
                matches.append(workload)

    if not matches:
        raise ValueError(
            f"No workload found for definition={definition!r}, batch_size={batch_size!r}"
        )
    if workload_index >= len(matches):
        raise IndexError(
            f"workload_index={workload_index} out of range for {len(matches)} matches"
        )
    return matches[workload_index]


def _load_tensor_input(dataset_path: Path, spec: dict[str, Any]) -> torch.Tensor:
    if spec["type"] != "safetensors":
        raise ValueError(f"Expected safetensors input, got {spec['type']!r}")
    tensor_path = (dataset_path / spec["path"]).resolve()
    tensors = load_file(str(tensor_path), device="cuda")
    return tensors[spec["tensor_key"]]


def load_workload_tensors(
    definition: str,
    *,
    batch_size: int | None = None,
    workload_index: int = 0,
) -> dict[str, Any]:
    """Load a GDN decode workload and allocate DPS outputs."""
    dataset_path = get_dataset_path()
    workload = load_workload_record(
        definition,
        batch_size=batch_size,
        workload_index=workload_index,
    )

    tensors: dict[str, Any] = {}
    for name, spec in workload["inputs"].items():
        if spec["type"] == "scalar":
            tensors[name] = spec["value"]
        elif spec["type"] == "safetensors":
            tensors[name] = _load_tensor_input(dataset_path, spec)
        else:
            raise ValueError(f"Unsupported input type for {name}: {spec['type']!r}")

    tensors["output"] = torch.empty_like(tensors["v"])
    tensors["new_state"] = torch.empty_like(tensors["state"])
    return tensors
