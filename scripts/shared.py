"""Shared helpers for local profiling scripts."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable


@dataclass(frozen=True)
class DEFS:
    DECODE: str = "gdn_decode_qk4_v8_d128_k_last"


ALGO_ENTRY_POINTS = {
    "fi-baseline": "kernel_fi_baseline",
    "fla-recurrent": "kernel_fla_recurrent",
    "fla-tma": "kernel_fla_tma",
}


def resolve_algo_names(algo: str) -> list[str]:
    """Resolve 'all' or a comma-separated list into canonical algorithm names."""
    if algo == "all":
        return list(ALGO_ENTRY_POINTS)

    names = [name.strip() for name in algo.split(",") if name.strip()]
    unknown = [name for name in names if name not in ALGO_ENTRY_POINTS]
    if unknown:
        valid = ", ".join(["all", *ALGO_ENTRY_POINTS])
        raise ValueError(f"Unknown algo(s): {unknown}. Valid choices: {valid}")
    return names


def load_algo_functions() -> dict[str, Callable]:
    """Load benchmarkable kernel functions from solution/triton/kernel.py."""
    kernel_module = import_module("solution.triton.kernel")
    return {
        algo_name: getattr(kernel_module, entry_point)
        for algo_name, entry_point in ALGO_ENTRY_POINTS.items()
    }
