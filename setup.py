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
import bisect
import fnmatch
import importlib
import io
import sys
from pathlib import Path
from typing import List, Tuple

REQUIRED = [
    ("torch",           "2.2.0"),
    ("transformers",    "4.47.0"),
    ("accelerate",      "0.30.0"),
    ("datasets",        "2.20.0"),
    ("peft",            "0.11.0"),
    ("trl",             "0.9.0"),
    ("yaml",            "6.0"),           # pip install pyyaml
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


class _MultiPartReader(io.RawIOBase):
    """
    Read-only, seekable view over split zip parts (image.zip.part-aa, -ab, …)
    as if they were one file. Lets zipfile read the archive in place, so the
    ~171 GB merged image.zip never has to be written to disk.
    """

    def __init__(self, parts: List[Path]):
        self._parts = parts
        self._sizes = [p.stat().st_size for p in parts]
        self._offsets = [0]
        for s in self._sizes:
            self._offsets.append(self._offsets[-1] + s)
        self._size = self._offsets[-1]
        self._pos = 0
        self._fh = None
        self._fh_idx = -1

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return pos

    def readinto(self, buf) -> int:
        view = memoryview(buf).cast("B")
        n_done = 0
        while n_done < len(view) and self._pos < self._size:
            idx = bisect.bisect_right(self._offsets, self._pos) - 1
            if idx != self._fh_idx:
                if self._fh:
                    self._fh.close()
                self._fh = open(self._parts[idx], "rb")
                self._fh_idx = idx
            local = self._pos - self._offsets[idx]
            self._fh.seek(local)
            want = min(len(view) - n_done, self._sizes[idx] - local)
            got = self._fh.readinto(view[n_done:n_done + want])
            if not got:
                break
            n_done += got
            self._pos += got
        return n_done

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
        super().close()


def _default_image_out_dir() -> str:
    """sft.local_image_dir from config.yaml, else ./data/llava_cot_images."""
    try:
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
        d = (cfg.get("sft") or {}).get("local_image_dir")
        if d:
            return str(d)
    except Exception:
        pass
    return "./data/llava_cot_images"


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


_EXTRACT_MARKER = ".llava_cot_extract_complete"


def setup_images(dataset_id: str, out_dir: str, delete_parts: bool = False) -> bool:
    """
    Download and extract images for Xkev/LLaVA-CoT-100k (or similar datasets
    that ship images as split zip archives in the HF repo).

    Steps:
      1. snapshot_download only the image.zip.part-* files (HF cache, local disk)
      2. Extract straight from the parts into <out_dir> (no merged image.zip)
         — <out_dir> may be on Google Drive; extraction is resumable.
      3. Optionally delete the downloaded parts (--delete-parts)
    """
    import shutil
    import zipfile

    print("── Image setup ──────────────────────────────────────")
    out_path = Path(out_dir).expanduser()
    print(f"  Dataset  : {dataset_id}")
    print(f"  Output   : {out_path}")

    if (out_path / _EXTRACT_MARKER).exists():
        print(f"  ✓  Images already extracted at {out_path} — skipping")
        return True

    if str(out_path).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        print("  ✗  Google Drive is not mounted. In a notebook cell run:")
        print("       from google.colab import drive; drive.mount('/content/drive')")
        return False

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("  ✗  huggingface_hub not installed")
        return False

    patterns = ["image.zip.part-*", "image.zip"]

    # Pre-flight: how big is the archive, and is there room for it?
    try:
        info = HfApi().dataset_info(dataset_id, files_metadata=True)
        archive_bytes = sum(
            (s.size or 0) for s in info.siblings
            if fnmatch.fnmatch(s.rfilename, patterns[0]) or s.rfilename == patterns[1]
        )
    except Exception:
        archive_bytes = 0
    if archive_bytes:
        print(f"  Archive  : {_gb(archive_bytes)} (download to local disk, "
              f"roughly the same again once extracted to the output dir)")
        out_probe = out_path
        while not out_probe.exists():
            out_probe = out_probe.parent
        free_out = shutil.disk_usage(out_probe).free
        if free_out < archive_bytes:
            print(f"  ⚠  Only {_gb(free_out)} reported free at {out_probe} — "
                  "extraction may run out of space")

    print("  Downloading zip parts (this may take a while; already-cached parts are reused) …")
    try:
        repo_path = Path(snapshot_download(
            dataset_id, repo_type="dataset", allow_patterns=patterns,
        ))
    except Exception as e:
        print(f"  ✗  snapshot_download failed: {e}")
        print("     If the local disk is full, delete a stale image.zip in the HF cache:")
        print("       rm -f ~/.cache/huggingface/hub/datasets--*/snapshots/*/image.zip")
        return False

    zip_parts = sorted(repo_path.glob("image.zip.part-*"))
    merged = repo_path / "image.zip"

    # Older versions of this script wrote a merged image.zip next to the parts;
    # if the parts exist that file is redundant (and possibly truncated).
    if zip_parts and merged.exists() and not merged.is_symlink():
        print(f"  Removing leftover merged archive {merged} ({_gb(merged.stat().st_size)})")
        merged.unlink()

    if zip_parts:
        print(f"  Reading {len(zip_parts)} zip parts in place (no merge needed)")
        source = _MultiPartReader(zip_parts)
    elif merged.exists():
        source = open(merged, "rb")
    else:
        print("  ✗  No image.zip or image.zip.part-* found in repo")
        return False

    print(f"  Extracting to {out_path} …")
    out_path.mkdir(parents=True, exist_ok=True)
    try:
        with source, zipfile.ZipFile(source) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            total = sum(m.file_size for m in members)
            done_bytes = skipped = 0
            try:
                from tqdm import tqdm
                bar = tqdm(total=total, unit="B", unit_scale=True, desc="  Extracting")
            except ImportError:
                bar = None
            for m in members:
                target = out_path / m.filename
                # Resume support: skip files already fully written
                if target.exists() and target.stat().st_size == m.file_size:
                    skipped += 1
                else:
                    zf.extract(m, out_path)
                done_bytes += m.file_size
                if bar:
                    bar.update(m.file_size)
            if bar:
                bar.close()
    except Exception as e:
        print(f"  ✗  Extraction failed: {e}")
        print("     Re-run the same command to resume — finished files are skipped.")
        return False

    (out_path / _EXTRACT_MARKER).write_text(f"{dataset_id}\n{len(members)} files\n")
    print(f"  ✓  {len(members)} files ready at {out_path}"
          + (f" ({skipped} already present)" if skipped else ""))

    if delete_parts:
        freed = 0
        for p in zip_parts + ([merged] if merged.exists() else []):
            real = p.resolve()          # HF cache snapshots are symlinks to blobs
            freed += real.stat().st_size
            real.unlink()
            if p.is_symlink():
                p.unlink()
        print(f"  ✓  Deleted downloaded zip parts, freed {_gb(freed)}")
    else:
        print("     Tip: pass --delete-parts to free the local disk used by the zip parts.")

    print(f"\n  config.yaml  →  sft.local_image_dir: {out_path}")
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
    p.add_argument("--image-out-dir", default=None,
                   help="Directory to extract images into "
                        "(default: sft.local_image_dir from config.yaml)")
    p.add_argument("--delete-parts", action="store_true",
                   help="Delete the downloaded zip parts after successful extraction")
    args = p.parse_args()

    run_image_setup = not args.exclude_setup_images and not args.check_only
    image_out_dir = args.image_out_dir or _default_image_out_dir()

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
        checks.append(("Image setup", setup_images(args.image_dataset, image_out_dir, args.delete_parts)))
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
