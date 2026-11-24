#!/usr/bin/env python3
"""
setup.py — Environment validation for Stage 4.

Run this once after installing dependencies to verify everything is in place
before running any experiment scripts.

Usage:
    python setup.py
    python setup.py --check-only       # skip HF auth test
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import List, Tuple

REQUIRED = [
    ("torch",           "2.2.0"),
    ("transformers",    "4.47.0"),
    ("accelerate",      "0.30.0"),
    ("datasets",        "2.20.0"),
    ("peft",            "0.11.0"),
    ("trl",             "0.9.0"),
    ("pyyaml",          "6.0"),
    ("huggingface_hub", "0.23.0"),
    ("wandb",           "0.17.0"),
    ("PIL",             None),          # Pillow — no version check
]


def _version_ok(installed: str, required: str) -> bool:
    from packaging.version import Version
    try:
        return Version(installed) >= Version(required)
    except Exception:
        return True  # can't parse → assume ok


def check_packages() -> bool:
    print("── Python packages ──────────────────────────────────")
    ok = True
    for pkg, req in REQUIRED:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            status = "✓"
            if req and ver != "?":
                if not _version_ok(ver, req):
                    status = f"⚠  (need ≥{req})"
                    ok = False
            print(f"  {status}  {pkg}  {ver}")
        except ImportError:
            print(f"  ✗  {pkg}  NOT FOUND")
            ok = False
    return ok


def check_python() -> bool:
    print("── Python version ───────────────────────────────────")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"  ✗  Python {major}.{minor} — need ≥ 3.10")
        return False
    print(f"  ✓  Python {major}.{minor}")
    return True


def check_gpu() -> bool:
    print("── GPU / CUDA ────────────────────────────────────────")
    try:
        import torch
        if not torch.cuda.is_available():
            print("  ⚠  No GPU detected — training will be very slow on CPU")
            return True  # not fatal
        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / 1e9
            flag = "✓" if vram >= 20 else "⚠ (<20 GB — may OOM on 7B model)"
            print(f"  {flag}  GPU {i}: {props.name}  {vram:.1f} GB VRAM")
        return True
    except Exception as e:
        print(f"  ✗  GPU check failed: {e}")
        return False


def check_config() -> bool:
    print("── config.yaml ──────────────────────────────────────")
    try:
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        required_keys = ["model", "lora", "sft", "eval", "generation", "seed"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            print(f"  ✗  Missing keys: {missing}")
            return False
        print(f"  ✓  config.yaml  (seed={cfg['seed']}, model={cfg['model']['name']})")
        return True
    except FileNotFoundError:
        print("  ✗  config.yaml not found — run from the repo root")
        return False
    except Exception as e:
        print(f"  ✗  Error reading config: {e}")
        return False


def check_hf_auth(check_only: bool) -> bool:
    print("── HuggingFace auth ─────────────────────────────────")
    if check_only:
        print("  –  Skipped (--check-only)")
        return True
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"  ✓  Logged in as: {user.get('name', '?')}")
        return True
    except Exception:
        print("  ⚠  Not authenticated — private datasets/models may fail")
        print("     Run: huggingface-cli login")
        return True  # not fatal for public datasets


def check_model_accessible(check_only: bool) -> bool:
    print("── Model accessibility ──────────────────────────────")
    if check_only:
        print("  –  Skipped (--check-only)")
        return True
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        model_name = cfg["model"]["name"]
        info = api.model_info(model_name)
        print(f"  ✓  {model_name}  accessible (gated={info.gated})")
        return True
    except Exception as e:
        print(f"  ⚠  Could not verify model access: {e}")
        return True  # non-fatal


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check-only", action="store_true",
                   help="Skip HF auth and model-access checks")
    args = p.parse_args()

    print("\n╔══ Stage 4 Environment Check ══════════════════════╗")
    checks: List[Tuple[str, bool]] = [
        ("Python version", check_python()),
        ("Packages",       check_packages()),
        ("GPU",            check_gpu()),
        ("config.yaml",    check_config()),
        ("HF auth",        check_hf_auth(args.check_only)),
        ("Model access",   check_model_accessible(args.check_only)),
    ]
    print("╚════════════════════════════════════════════════════╝\n")

    all_ok = all(ok for _, ok in checks)
    if all_ok:
        print("✓  All checks passed — ready to run experiments.")
    else:
        failed = [name for name, ok in checks if not ok]
        print(f"✗  Issues found in: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
