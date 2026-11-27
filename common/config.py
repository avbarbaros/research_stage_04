"""
common/config.py — Configuration loading, run-directory management, and seed setting.

Follows the same interface as Stage 3 so cross-stage comparisons stay easy.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml


# ── Loading ───────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml", overrides: List[str] = ()) -> Dict[str, Any]:
    """
    Load YAML config and apply optional CLI overrides.

    Override syntax: "key.nested.path=value"
    Values are parsed with yaml.safe_load, so booleans, ints, floats and
    strings all work as expected.

    Example:
        cfg = load_config("config.yaml", overrides=["sft.max_samples=1000"])
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override '{ov}' has no '='; expected key.path=value")
        key_path, raw_value = ov.split("=", 1)
        value = yaml.safe_load(raw_value)
        _apply_override(cfg, key_path.strip(), value)

    return cfg


def _apply_override(cfg: dict, key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


# ── Run directories ───────────────────────────────────────────────────────────

def make_run_dir(output_root: str, experiment_key: str, run_name: str) -> Path:
    """
    Create and return  <output_root>/<experiment_key>/<run_name>_<timestamp>/.

    The timestamp ensures multiple runs never clash.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / experiment_key / f"{run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_resolved_config(cfg: Dict[str, Any], run_dir: Path) -> None:
    """Persist the fully-resolved config alongside the run outputs."""
    with open(run_dir / "config_resolved.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Environment info ──────────────────────────────────────────────────────────

def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import peft
        info["peft"] = peft.__version__
    except ImportError:
        pass
    return info


# ── Argument parser helper ────────────────────────────────────────────────────

def base_argparser(description: str = "") -> "argparse.ArgumentParser":
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument(
        "--set",
        nargs="*",
        action="extend",          # repeated --set flags accumulate (default "store" keeps only the last)
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set sft.max_samples=1000",
    )
    p.add_argument("--fresh", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return p
