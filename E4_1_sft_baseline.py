#!/usr/bin/env python3
"""
E4.1 — SFT Baseline

Supervised fine-tune Qwen2.5-VL-7B-Instruct on a multimodal Chain-of-Thought
dataset, then evaluate on MathVista, MathVision, and MMMU.

This establishes the documented, reproducible baseline against which ALL
subsequent RL claims (E4.2 GRPO, E4.4 reward surgery, …) are measured.
Treat the numbers from this script as the official reference; never compare
an RL result against the base model directly.

Usage
-----
  # Full run (train then eval)
  python E4_1_sft_baseline.py

  # Quick smoke-test (1k training samples, 100 eval items each)
  python E4_1_sft_baseline.py --set sft.max_samples=1000 \\
      --set eval.mathvista.max_samples=100 \\
      --set eval.mathvision.max_samples=100 \\
      --set eval.mmmu.max_samples=100

  # Eval only (from a saved checkpoint)
  python E4_1_sft_baseline.py --mode eval --checkpoint outputs/e4_1/…/adapter_final

  # Re-analyze already-generated JSONL without re-running inference
  python E4_1_sft_baseline.py --mode analyze --run-dir outputs/e4_1/…

Colab L4 GPU (22.5 GB VRAM) notes
-----------------------------------
  Qwen2.5-VL-7B in bfloat16  ≈ 14 GB
  LoRA overhead               ≈  0.1 GB
  Activations (gc, bs=2)      ≈  4–5 GB
  Total                       ≈ 18–20 GB  → fits with gradient checkpointing

  If you hit OOM, add:  --set sft.per_device_train_batch_size=1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Reduce CUDA memory fragmentation (must be set before torch initialises CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

# ── Local imports ─────────────────────────────────────────────────────────────
from common.config import (
    load_config,
    make_run_dir,
    save_resolved_config,
    set_seed,
    environment_info,
    base_argparser,
)
from common.dataset_loaders import (
    load_image,
    load_sft_dataset,
    load_mathvista,
    load_mathvision,
    load_mmmu,
)
from common.scoring_logic import (
    score_mathvista,
    score_mathvision,
    score_mmmu,
    accuracy_report,
    wilson_ci,
)
from common.collator import VLMSFTCollator


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = base_argparser(
        "E4.1 SFT Baseline — fine-tune Qwen2.5-VL and evaluate on MathVista/MathVision/MMMU"
    )
    p.add_argument(
        "--mode",
        choices=["train", "eval", "train_eval", "analyze"],
        default="train_eval",
        help=(
            "train       : fine-tuning only\n"
            "eval        : inference + scoring (requires --checkpoint)\n"
            "train_eval  : fine-tuning then evaluation (default)\n"
            "analyze     : re-score existing JSONL outputs (requires --run-dir)"
        ),
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to saved LoRA adapter directory (for --mode eval)",
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="Existing run directory (for --mode analyze)",
    )
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def _get_model_class():
    """
    Return the correct VLM class for the installed transformers version.

    Try class names from newest to oldest so there is no cross-version warning:
      1. Qwen2_5_VLForConditionalGeneration  (transformers ≥ 4.52, matches qwen2_5_vl config)
      2. Qwen2VLForConditionalGeneration      (older transformers, still works with a warning)
    """
    import transformers
    for class_name in (
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
    ):
        cls = getattr(transformers, class_name, None)
        if cls is not None:
            print(f"[model] Using model class: {class_name}")
            return cls
    raise ImportError(
        "Could not find Qwen2_5_VLForConditionalGeneration or "
        "Qwen2VLForConditionalGeneration in the installed transformers package. "
        f"Installed version: {transformers.__version__}"
    )


def load_base_model(cfg: Dict) -> tuple:
    """Load Qwen2.5-VL and its processor."""
    from transformers import AutoProcessor

    model_name = cfg["model"]["name"]
    dtype = getattr(torch, cfg["model"]["dtype"])
    use_fa2 = cfg["model"].get("use_flash_attn", False)

    # Determine attention implementation, with automatic fallback
    if use_fa2:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            print("[model] flash-attn not found — falling back to sdpa")
            attn_impl = "sdpa"
    else:
        attn_impl = "sdpa"

    print(f"[model] Loading {model_name}  dtype={cfg['model']['dtype']}  attn={attn_impl}")

    # Cap image resolution: Qwen2.5-VL's default max_pixels (~12.8 MP) turns a
    # large document/chart image into >16k visual tokens (RAM/VRAM blow-up and
    # truncation of image tokens). Each 28×28 patch = 1 visual token.
    processor = AutoProcessor.from_pretrained(model_name)
    min_px, max_px = cfg["model"].get("min_pixels"), cfg["model"].get("max_pixels")
    if min_px or max_px:
        # transformers ≥5 API: size={"shortest_edge": min_pixels, "longest_edge":
        # max_pixels} (both are pixel *counts* for Qwen2-VL image processors).
        ip = processor.image_processor
        size = dict(ip.size or {})
        if min_px:
            size["shortest_edge"] = int(min_px)
        if max_px:
            size["longest_edge"] = int(max_px)
        ip.size = size
        for attr, val in (("min_pixels", min_px), ("max_pixels", max_px)):
            if val and hasattr(ip, attr):        # older versions read these
                setattr(ip, attr, int(val))
        print(f"[model] Processor image limits: {size}")

    ModelClass = _get_model_class()
    model = ModelClass.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    model.config.use_cache = False          # required for gradient checkpointing

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[model] Loaded  {n_params:.1f}B parameters")
    return model, processor


def apply_lora(model: Any, cfg: Dict) -> Any:
    """Wrap model with LoRA adapters (language model layers only)."""
    from peft import LoraConfig, TaskType, get_peft_model

    lc = cfg["lora"]
    lora_config = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        target_modules=lc["target_modules"],
        bias=lc.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── SFT training ──────────────────────────────────────────────────────────────

class _SFTDataset(Dataset):
    """Holds undecoded image references; decodes one image per __getitem__."""

    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        # Fall through to the next example if an image turns out to be corrupt.
        for k in range(len(self.examples)):
            ex = self.examples[(idx + k) % len(self.examples)]
            image = load_image(ex["image"])
            if image is not None:
                return {**ex, "image": image}
        raise RuntimeError("No decodable images in the SFT dataset")


def run_training(cfg: Dict, model: Any, processor: Any, run_dir: Path) -> Path:
    """
    Fine-tune model using HuggingFace Trainer + custom VLM collator.

    Returns the path to the saved final LoRA adapter.
    """
    from transformers import TrainingArguments, Trainer

    sc = cfg["sft"]

    print("[train] Loading SFT dataset …")
    examples = load_sft_dataset(
        dataset_id=sc["dataset_id"],
        max_samples=sc.get("max_samples"),
        seed=cfg["seed"],
        local_image_dir=sc.get("local_image_dir"),
    )
    dataset = _SFTDataset(examples)
    collator = VLMSFTCollator(processor=processor, max_seq_len=sc["max_seq_len"])

    checkpoints_dir = run_dir / sc.get("checkpoints_subdir", "checkpoints")

    # Compute warmup_steps manually — warmup_ratio was removed in transformers 5.x
    total_steps = max(1, (
        len(dataset)
        // (sc["per_device_train_batch_size"] * sc["gradient_accumulation_steps"])
    ) * sc["num_train_epochs"])
    warmup_steps = max(1, int(total_steps * sc.get("warmup_ratio", 0.03)))

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=sc["num_train_epochs"],
        per_device_train_batch_size=sc["per_device_train_batch_size"],
        gradient_accumulation_steps=sc["gradient_accumulation_steps"],
        learning_rate=sc["learning_rate"],
        lr_scheduler_type=sc.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        weight_decay=sc.get("weight_decay", 0.01),
        max_grad_norm=sc.get("max_grad_norm", 1.0),
        bf16=True,
        logging_steps=sc.get("logging_steps", 50),
        save_steps=sc.get("save_steps", 500),
        save_total_limit=sc.get("save_total_limit", 2),
        dataloader_num_workers=sc.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb" if not cfg.get("_no_wandb") else "none",
        run_name=cfg.get("wandb", {}).get("run_name", "e4_1_sft"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("[train] Starting supervised fine-tuning …")
    trainer.train()

    # Save final LoRA adapter
    adapter_path = run_dir / sc.get("adapter_subdir", "adapter_final")
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    print(f"[train] Adapter saved → {adapter_path}")
    return adapter_path


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_inference(
    model: Any,
    processor: Any,
    benchmark_name: str,
    examples: List[Dict],
    cfg: Dict,
    run_dir: Path,
) -> Path:
    """
    Run model inference on a benchmark, writing results to JSONL.

    Resumable: already-completed item IDs are loaded from an existing JSONL
    file and skipped. Run with --fresh to restart from scratch.
    """
    gc = cfg["generation"]
    out_file = run_dir / f"eval_{benchmark_name}.jsonl"

    # Resumption: collect IDs already written
    done_ids: set = set()
    if out_file.exists():
        with open(out_file) as fh:
            for line in fh:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    todo = [ex for ex in examples if ex["id"] not in done_ids]
    total = len(examples)
    skipped = total - len(todo)
    print(f"[eval:{benchmark_name}] {len(todo)} items to run  ({skipped} skipped)")

    if not todo:
        return out_file

    model.eval()
    device = next(model.parameters()).device

    with open(out_file, "a", buffering=1) as fh:
        t0 = time.time()
        for i, ex in enumerate(todo):
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(todo) - i) / rate if rate > 0 else 0
                print(f"[eval:{benchmark_name}]  {i}/{len(todo)}  "
                      f"{rate:.1f} items/s  ETA {eta/60:.1f} min")

            image = load_image(ex["image"])     # decode on demand (RAM)
            if image is None:
                print(f"[eval:{benchmark_name}] ⚠ could not decode image for {ex['id']} — skipped")
                continue

            # Build prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": ex["question"]},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=gc["max_new_tokens"],
                    temperature=gc.get("temperature", 0.0),
                    do_sample=gc.get("do_sample", False),
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][prompt_len:]
            response = processor.decode(new_tokens, skip_special_tokens=True).strip()

            row = {
                "id": ex["id"],
                "question": ex["question"],
                "response": response,
                "gt_answer": ex["answer"],
                "metadata": ex.get("metadata", {}),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[eval:{benchmark_name}] Done in {(time.time()-t0)/60:.1f} min")
    return out_file


def analyze_jsonl(out_file: Path, scorer_fn) -> Dict:
    """Load a JSONL result file and compute accuracy metrics."""
    rows = []
    with open(out_file) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    correct = 0
    details = []
    for row in rows:
        ok = scorer_fn(row["response"], row["gt_answer"], row.get("metadata", {}))
        correct += int(ok)
        details.append({**row, "correct": ok})

    total = len(rows)
    acc = correct / total if total else 0.0
    lo, hi = wilson_ci(correct, total)

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "ci_lo": lo,
        "ci_hi": hi,
        "details": details,
    }


# ── Results summary ───────────────────────────────────────────────────────────

def print_and_save_summary(results: Dict, run_dir: Path, cfg: Dict) -> None:
    summary = {
        "run_dir": str(run_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": cfg["model"]["name"],
        "sft_dataset": cfg["sft"]["dataset_id"],
        "sft_max_samples": cfg["sft"].get("max_samples"),
        "results": {
            k: {
                "accuracy": v["accuracy"],
                "correct": v["correct"],
                "total": v["total"],
                "ci_95": [round(v["ci_lo"], 4), round(v["ci_hi"], 4)],
            }
            for k, v in results.items()
        },
    }

    out_path = run_dir / "summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 60)
    print("E4.1 SFT BASELINE — RESULTS")
    print("=" * 60)
    for bench, res in results.items():
        print(accuracy_report(f"  {bench:12s}", res["correct"], res["total"]))
    print(f"\n  Run dir  : {run_dir}")
    print(f"  Summary  : {out_path}")
    print("=" * 60)


# ── W&B logging ───────────────────────────────────────────────────────────────

def wandb_log_results(results: Dict, cfg: Dict) -> None:
    try:
        import wandb
        for bench, res in results.items():
            wandb.log({
                f"eval/{bench}/accuracy": res["accuracy"],
                f"eval/{bench}/correct": res["correct"],
                f"eval/{bench}/total": res["total"],
            })
    except Exception as e:
        print(f"[wandb] Logging failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set or [])
    cfg["_no_wandb"] = args.no_wandb

    set_seed(cfg["seed"])

    env = environment_info()
    print("[env]", json.dumps(env, indent=2))

    # ── Run directory ──────────────────────────────────────────────────────
    if args.mode == "analyze":
        if not args.run_dir:
            sys.exit("[error] --mode analyze requires --run-dir")
        run_dir = Path(args.run_dir)
    else:
        run_dir = make_run_dir(
            cfg["output_root"],
            "e4_1",
            cfg.get("wandb", {}).get("run_name", "sft_baseline"),
        )
        save_resolved_config(cfg, run_dir)
    print(f"[main] Run directory: {run_dir}")

    # ── W&B init ───────────────────────────────────────────────────────────
    if not args.no_wandb and args.mode != "analyze":
        try:
            import wandb
            wandb.init(
                project=cfg.get("wandb", {}).get("project", "research_stage_04"),
                name=cfg.get("wandb", {}).get("run_name", "e4_1_sft_baseline"),
                config=cfg,
            )
        except Exception as e:
            print(f"[wandb] Init failed (continuing without W&B): {e}")
            cfg["_no_wandb"] = True

    # ── Training ───────────────────────────────────────────────────────────
    if args.mode in ("train", "train_eval"):
        model, processor = load_base_model(cfg)
        model = apply_lora(model, cfg)
        adapter_path = run_training(cfg, model, processor, run_dir)

    # ── Load checkpoint for eval ───────────────────────────────────────────
    if args.mode in ("eval", "train_eval"):
        if args.mode == "eval":
            # Fresh load — no LoRA applied yet
            model, processor = load_base_model(cfg)
            checkpoint = args.checkpoint or str(run_dir / cfg["sft"].get("adapter_subdir", "adapter_final"))
            print(f"[eval] Loading LoRA adapter from {checkpoint}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, checkpoint)
        # (if train_eval, model already has LoRA loaded from training)

        results: Dict = {}

        # ── MathVista ──────────────────────────────────────────────────────
        mv_examples = load_mathvista(
            cfg["eval"]["mathvista"]["dataset_id"],
            cfg["eval"]["mathvista"]["split"],
            cfg["eval"]["mathvista"].get("max_samples"),
            cfg["seed"],
        )
        mv_outfile = run_inference(model, processor, "mathvista", mv_examples, cfg, run_dir)
        results["mathvista"] = analyze_jsonl(mv_outfile, score_mathvista)

        # ── MathVision ─────────────────────────────────────────────────────
        mvision_examples = load_mathvision(
            cfg["eval"]["mathvision"]["dataset_id"],
            cfg["eval"]["mathvision"]["split"],
            cfg["eval"]["mathvision"].get("max_samples"),
            cfg["seed"],
        )
        mvision_outfile = run_inference(model, processor, "mathvision", mvision_examples, cfg, run_dir)
        results["mathvision"] = analyze_jsonl(mvision_outfile, score_mathvision)

        # ── MMMU ───────────────────────────────────────────────────────────
        mmmu_examples = load_mmmu(
            cfg["eval"]["mmmu"]["dataset_id"],
            cfg["eval"]["mmmu"]["split"],
            cfg["eval"]["mmmu"].get("max_samples"),
            cfg["seed"],
            subjects=cfg["eval"]["mmmu"].get("subjects"),
        )
        mmmu_outfile = run_inference(model, processor, "mmmu", mmmu_examples, cfg, run_dir)
        results["mmmu"] = analyze_jsonl(mmmu_outfile, score_mmmu)

        print_and_save_summary(results, run_dir, cfg)

        if not cfg.get("_no_wandb"):
            wandb_log_results(results, cfg)

    # ── Analyze only (re-score existing JSONL) ─────────────────────────────
    elif args.mode == "analyze":
        SCORERS = {
            "mathvista": score_mathvista,
            "mathvision": score_mathvision,
            "mmmu": score_mmmu,
        }
        results = {}
        for bench, scorer in SCORERS.items():
            out_file = run_dir / f"eval_{bench}.jsonl"
            if not out_file.exists():
                print(f"[analyze] {out_file} not found — skipping")
                continue
            results[bench] = analyze_jsonl(out_file, scorer)

        if results:
            print_and_save_summary(results, run_dir, cfg)
        else:
            print("[analyze] No JSONL files found.")

    # ── Finish ─────────────────────────────────────────────────────────────
    if not cfg.get("_no_wandb"):
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    print("[main] Done.")


if __name__ == "__main__":
    main()
