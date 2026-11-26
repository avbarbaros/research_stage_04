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
    as if they were one file, so zipfile can read the archive without a merged
    image.zip ever being written.

    Parts are fetched lazily through ``fetch(idx) -> Path`` the first time a
    byte inside them is read, and can be dropped again with ``release(idx)``.
    Together this lets the caller keep only a few parts on disk at a time.
    """

    def __init__(self, sizes: List[int], fetch):
        self._sizes = list(sizes)
        self._offsets = [0]
        for s in self._sizes:
            self._offsets.append(self._offsets[-1] + s)
        self._size = self._offsets[-1]
        self._fetch = fetch
        self._paths: dict = {}      # idx -> local Path of fetched parts
        self._pos = 0
        self._fh = None
        self._fh_idx = -1

    # ── part bookkeeping ──
    def part_index(self, pos: int) -> int:
        return bisect.bisect_right(self._offsets, pos) - 1

    def part_end(self, idx: int) -> int:
        return self._offsets[idx + 1]

    def loaded(self) -> List[int]:
        return sorted(self._paths)

    def release(self, idx: int):
        """Forget part idx (closing its handle); returns its path or None."""
        if idx == self._fh_idx:
            self._fh.close()
            self._fh, self._fh_idx = None, -1
        return self._paths.pop(idx, None)

    # ── file-object protocol ──
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
            idx = self.part_index(self._pos)
            if idx not in self._paths:
                self._paths[idx] = Path(self._fetch(idx))
            if idx != self._fh_idx:
                if self._fh:
                    self._fh.close()
                self._fh = open(self._paths[idx], "rb")
                self._fh_idx = idx
            local = self._pos - self._offsets[idx]
            self._fh.seek(local)
            want = min(len(view) - n_done, self._sizes[idx] - local)
            got = self._fh.readinto(view[n_done:n_done + want])
            if not got:
                raise IOError(f"unexpected end of zip part {self._paths[idx]}")
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


def _delete_file(p: Path) -> int:
    """Delete p (following an HF-cache symlink to its blob); returns bytes freed."""
    real = p.resolve()
    freed = real.stat().st_size if real.exists() else 0
    real.unlink(missing_ok=True)
    if p.is_symlink():
        p.unlink()
    return freed


_EXTRACT_MARKER = ".llava_cot_extract_complete"


def setup_images(dataset_id: str, out_dir: str, parts_dir: str,
                 keep_parts: bool = False) -> bool:
    """
    Download and extract images for Xkev/LLaVA-CoT-100k (or similar datasets
    that ship images as split zip archives in the HF repo), using little
    local disk.

    The parts are byte-slices of ONE zip (a file can straddle two parts and
    the zip index lives in the last part), so they can't be unzipped one by
    one. Instead:
      1. Download the part(s) holding the zip index (end of the archive).
      2. Extract files in on-disk order into <out_dir> (may be Google Drive),
         downloading each part only when extraction reaches it.
      3. Delete each part as soon as every file stored in it is extracted
         (unless --keep-parts). Peak local usage ≈ 2–3 parts (~10 GB each).
    Re-running resumes: finished files are skipped, and parts that contain
    only finished files are never downloaded again.
    """
    import shutil
    import zipfile

    print("── Image setup ──────────────────────────────────────")
    out_path = Path(out_dir).expanduser()
    parts_path = Path(parts_dir).expanduser()
    print(f"  Dataset  : {dataset_id}")
    print(f"  Output   : {out_path}")
    print(f"  Parts dir: {parts_path}  (temporary, local disk)")

    if (out_path / _EXTRACT_MARKER).exists():
        print(f"  ✓  Images already extracted at {out_path} — skipping")
        return True

    if str(out_path).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        print("  ✗  Google Drive is not mounted. In a notebook cell run:")
        print("       from google.colab import drive; drive.mount('/content/drive')")
        return False

    try:
        from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        print("  ✗  huggingface_hub not installed")
        return False

    # ── list archive parts and their sizes (needed to map offsets → parts) ──
    try:
        info = HfApi().dataset_info(dataset_id, files_metadata=True)
        repo_files = {s.rfilename: s.size for s in info.siblings}
    except Exception as e:
        print(f"  ✗  Could not list files of {dataset_id}: {e}")
        return False
    part_names = sorted(f for f in repo_files if fnmatch.fnmatch(f, "image.zip.part-*"))
    if not part_names and "image.zip" in repo_files:
        part_names = ["image.zip"]
    if not part_names:
        print("  ✗  No image.zip or image.zip.part-* found in repo")
        return False
    sizes = [repo_files[n] for n in part_names]
    if any(s is None for s in sizes):
        print("  ✗  Hub did not report part sizes; cannot stream the archive")
        return False
    print(f"  Archive  : {_gb(sum(sizes))} in {len(part_names)} part(s), "
          f"largest {_gb(max(sizes))}")

    # Leftover merged image.zip written by older versions of this script
    # into the HF cache (up to ~171 GB of wasted local disk).
    if part_names[0] != "image.zip":
        repo_cache = Path(HF_HUB_CACHE) / f"datasets--{dataset_id.replace('/', '--')}"
        for stale in repo_cache.glob("snapshots/*/image.zip"):
            if not stale.is_symlink():
                print(f"  Removing leftover merged archive {stale} ({_gb(stale.stat().st_size)})")
                stale.unlink()

    out_probe = out_path
    while not out_probe.exists():
        out_probe = out_probe.parent
    if shutil.disk_usage(out_probe).free < sum(sizes):
        print(f"  ⚠  Only {_gb(shutil.disk_usage(out_probe).free)} reported free at "
              f"{out_probe} — extraction may run out of space")

    parts_path.mkdir(parents=True, exist_ok=True)
    stats = {"downloaded": 0, "freed": 0}

    def fetch(idx: int) -> Path:
        name, size = part_names[idx], sizes[idx]
        local = parts_path / name
        if local.exists() and local.stat().st_size == size:
            return local                                   # from an earlier run
        cached = try_to_load_from_cache(dataset_id, name, repo_type="dataset")
        if isinstance(cached, str) and Path(cached).exists() \
                and Path(cached).resolve().stat().st_size == size:
            return Path(cached)                            # from the HF cache
        free = shutil.disk_usage(parts_path).free
        if free < size:
            raise OSError(f"need {_gb(size)} for {name} but only {_gb(free)} free "
                          f"in {parts_path}")
        print(f"\n  ↓  Downloading {name} ({_gb(size)}, part {idx + 1}/{len(part_names)}) …")
        stats["downloaded"] += 1
        return Path(hf_hub_download(dataset_id, name, repo_type="dataset",
                                    local_dir=str(parts_path)))

    def release(idx: int) -> None:
        p = reader.release(idx)
        if p is None or keep_parts:
            return
        stats["freed"] += _delete_file(p)
        print(f"\n  🗑  Deleted {part_names[idx]} (fully extracted)")

    print(f"  Extracting to {out_path} …")
    out_path.mkdir(parents=True, exist_ok=True)
    reader = _MultiPartReader(sizes, fetch)
    try:
        with reader, zipfile.ZipFile(reader) as zf:
            # Parts holding the zip index must stay until the end.
            pinned = set(range(reader.part_index(zf.start_dir), len(part_names)))
            members = sorted((m for m in zf.infolist() if not m.is_dir()),
                             key=lambda m: m.header_offset)
            skipped = 0
            try:
                from tqdm import tqdm
                bar = tqdm(total=sum(m.file_size for m in members),
                           unit="B", unit_scale=True, desc="  Extracting")
            except ImportError:
                bar = None
            for m in members:
                # Every part ending before this file starts is fully extracted.
                for idx in reader.loaded():
                    if idx not in pinned and reader.part_end(idx) <= m.header_offset:
                        release(idx)
                target = out_path / m.filename
                if target.exists() and target.stat().st_size == m.file_size:
                    skipped += 1                          # resume: already done
                else:
                    zf.extract(m, out_path)
                if bar:
                    bar.update(m.file_size)
            if bar:
                bar.close()
            for idx in reader.loaded():
                release(idx)
    except Exception as e:
        print(f"\n  ✗  Extraction failed: {e}")
        print("     Re-run the same command to resume — finished files are skipped and")
        print(f"     parts already in {parts_path} are reused.")
        return False

    (out_path / _EXTRACT_MARKER).write_text(f"{dataset_id}\n{len(members)} files\n")
    print(f"  ✓  {len(members)} files ready at {out_path}"
          + (f" ({skipped} already present)" if skipped else ""))
    print(f"     Parts downloaded: {stats['downloaded']}, local disk freed: {_gb(stats['freed'])}")
    if not keep_parts:
        shutil.rmtree(parts_path / ".cache", ignore_errors=True)   # hf_hub_download metadata
        try:
            parts_path.rmdir()
        except OSError:
            pass
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
    p.add_argument("--parts-dir", default="./data/_image_zip_parts",
                   help="Local scratch dir for zip parts while extracting "
                        "(default: ./data/_image_zip_parts)")
    p.add_argument("--keep-parts", action="store_true",
                   help="Keep each zip part instead of deleting it once extracted")
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
        checks.append(("Image setup", setup_images(args.image_dataset, image_out_dir, args.parts_dir, args.keep_parts)))
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
