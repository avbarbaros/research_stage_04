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
    # torchao is an optional PEFT dependency; if present it must be ≥ 0.16.0
    # (peft 0.21+ raises ImportError if torchao < 0.16.0 is installed)
    ("torchao",         "0.16.0"),
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


def setup_images(dataset_id: str, out_dir: str) -> bool:
    """
    Download and extract images for Xkev/LLaVA-CoT-100k (or similar datasets
    that ship images as split zip archives in the HF repo).

    Steps:
      1. snapshot_download — fetches the full repo (JSONL + zip parts)
      2. Merge image.zip.part-* → image.zip
      3. unzip → <out_dir>/
      4. Print the path to paste into config.yaml as sft.local_image_dir
    """
    import subprocess
    import zipfile

    print("── Image setup ──────────────────────────────────────")
    print(f"  Dataset  : {dataset_id}")
    print(f"  Output   : {out_dir}")
    print("  Downloading repo (this may take a while) …")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ✗  huggingface_hub not installed")
        return False

    try:
        repo_path = importlib.import_module("pathlib").Path(
            snapshot_download(dataset_id, repo_type="dataset")
        )
    except Exception as e:
        print(f"  ✗  snapshot_download failed: {e}")
        return False

    # Merge split zip parts if present
    zip_parts = sorted(repo_path.glob("image.zip.part-*"))
    combined = repo_path / "image.zip"

    if zip_parts and not combined.exists():
        print(f"  Merging {len(zip_parts)} zip parts …")
        try:
            with open(combined, "wb") as fout:
                for part in zip_parts:
                    fout.write(part.read_bytes())
        except Exception as e:
            print(f"  ✗  Merge failed: {e}")
            return False

    if not combined.exists():
        print("  ✗  No image.zip or image.zip.part-* found in repo")
        return False

    # Extract
    out_path = importlib.import_module("pathlib").Path(out_dir)
    if out_path.exists() and any(out_path.iterdir()):
        print(f"  ✓  {out_path} already exists and is non-empty — skipping extraction")
    else:
        print(f"  Extracting to {out_path} …")
        out_path.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(combined) as zf:
                zf.extractall(out_path)
        except Exception as e:
            print(f"  ✗  Extraction failed: {e}")
            return False

    print(f"  ✓  Images ready at: {out_path}")
    print(f"\n  Add this to config.yaml  →  sft.local_image_dir: {out_path}")
    print(f"  Or pass on the CLI      →  --set sft.local_image_dir={out_path}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage 4 environment check + one-time image setup",
    )
    p.add_argument("--check-only", action="store_true",
                   help="Skip HF auth, model-access, and image-setup steps")
    p.add_argument("--exclude-setup-images", action="store_true",
                   help="Skip the image download/extraction step")
    p.add_argument("--image-dataset", default="Xkev/LLaVA-CoT-100k",
                   help="HF dataset ID to download images from (default: Xkev/LLaVA-CoT-100k)")
    p.add_argument("--image-out-dir", default="./data/llava_cot_images",
                   help="Local directory to extract images into")
    args = p.parse_args()

    run_image_setup = not args.exclude_setup_images and not args.check_only

    print("\n╔══ Stage 4 Environment Check ══════════════════════╗")
    checks: List[Tuple[str, bool]] = [
        ("Python version", check_python()),
        ("Packages",       check_packages()),
        ("GPU",            check_gpu()),
        ("config.yaml",    check_config()),
        ("HF auth",        check_hf_auth(args.check_only)),
        ("Model access",   check_model_accessible(args.check_only)),
    ]
    if run_image_setup:
        checks.append(("Image setup", setup_images(args.image_dataset, args.image_out_dir)))
    else:
        print("── Image setup ──────────────────────────────────────")
        print("  –  Skipped (--exclude-setup-images or --check-only)")
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
