"""
common/dataset_loaders.py — Dataset loading for SFT training and evaluation benchmarks.

SFT source  : lmms-lab/LLaVA-CoT-100k  (configurable)
Eval sources: AI4Math/MathVista, MathLLMs/MathVision, MMMU/MMMU
"""

from __future__ import annotations

import io
import random
from typing import Any, Dict, List, Optional

from PIL import Image


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_pil(image_field: Any) -> Optional[Image.Image]:
    """Convert whatever the HF dataset gives us into a PIL Image."""
    if image_field is None:
        return None
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if isinstance(image_field, bytes):
        return Image.open(io.BytesIO(image_field)).convert("RGB")
    if isinstance(image_field, dict) and "bytes" in image_field:
        return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
    return None


def _select_subset(items: list, max_samples: Optional[int], seed: int) -> list:
    if max_samples is None or max_samples >= len(items):
        return items
    rng = random.Random(seed)
    return rng.sample(items, max_samples)


# ── SFT dataset ───────────────────────────────────────────────────────────────

def load_sft_dataset(
    dataset_id: str,
    max_samples: Optional[int],
    seed: int,
) -> List[Dict]:
    """
    Load a multimodal CoT dataset for supervised fine-tuning.

    Returns a list of dicts with keys:
        id       : str
        image    : PIL.Image
        question : str   (user turn, without image placeholder tokens)
        answer   : str   (assistant turn — full CoT trace)

    Supported dataset formats:
      1. LLaVA conversations format  {"conversations": [{"from":"human",...}, {"from":"gpt",...}]}
      2. LLaVA-CoT format            {"problem": ..., "solution": ...}
      3. Generic instruction format  {"question": ..., "answer": ...}
    """
    from datasets import load_dataset

    print(f"[dataset] Loading SFT dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="train", trust_remote_code=True)
    print(f"[dataset] Raw size: {len(ds)}")

    examples = []
    for i, row in enumerate(ds):
        ex = _parse_sft_row(row, i)
        if ex is not None:
            examples.append(ex)

    print(f"[dataset] Parsed {len(examples)} valid examples")
    examples = _select_subset(examples, max_samples, seed)
    print(f"[dataset] Using {len(examples)} examples (max_samples={max_samples})")
    return examples


def _parse_sft_row(row: dict, idx: int) -> Optional[Dict]:
    """Parse one row from an SFT dataset into the normalised format."""
    image = _to_pil(row.get("image"))
    if image is None:
        return None  # skip text-only rows

    # Format 1: LLaVA conversations
    if "conversations" in row:
        convs = row["conversations"]
        human = next((c for c in convs if c.get("from") in ("human", "user")), None)
        gpt = next((c for c in convs if c.get("from") in ("gpt", "assistant")), None)
        if human is None or gpt is None:
            return None
        question = human["value"].replace("<image>", "").replace("<image>\n", "").strip()
        answer = gpt["value"].strip()
        if not question or not answer:
            return None
        return {
            "id": row.get("id", str(idx)),
            "image": image,
            "question": question,
            "answer": answer,
        }

    # Format 2: LLaVA-CoT / problem-solution
    if "problem" in row and "solution" in row:
        question = str(row["problem"]).strip()
        answer = str(row["solution"]).strip()
        if not question or not answer:
            return None
        return {
            "id": row.get("id", str(idx)),
            "image": image,
            "question": question,
            "answer": answer,
        }

    # Format 3: generic question-answer
    if "question" in row and "answer" in row:
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        if not question or not answer:
            return None
        return {
            "id": row.get("id", str(idx)),
            "image": image,
            "question": question,
            "answer": answer,
        }

    return None


# ── Evaluation datasets ───────────────────────────────────────────────────────

def load_mathvista(
    dataset_id: str,
    split: str,
    max_samples: Optional[int],
    seed: int,
) -> List[Dict]:
    """
    Load MathVista testmini.

    Returns dicts with keys:
        id, image, question, answer, metadata
    metadata includes: question_type, answer_type, choices
    """
    from datasets import load_dataset

    print(f"[dataset] Loading MathVista ({split})...")
    ds = load_dataset(dataset_id, split=split, trust_remote_code=True)

    examples = []
    for row in ds:
        image = _to_pil(row.get("decoded_image") or row.get("image"))
        if image is None:
            continue

        # Build question text
        question_parts = []
        if row.get("question"):
            question_parts.append(row["question"])
        if row.get("choices"):
            choices = row["choices"]
            if isinstance(choices, list):
                lettered = [f"({chr(65+i)}) {c}" for i, c in enumerate(choices)]
                question_parts.append("Choices: " + "  ".join(lettered))

        question = "\n".join(question_parts)
        answer = str(row.get("answer", "")).strip()

        examples.append({
            "id": str(row.get("pid", row.get("id", len(examples)))),
            "image": image,
            "question": question,
            "answer": answer,
            "metadata": {
                "question_type": row.get("question_type", ""),
                "answer_type": row.get("answer_type", ""),
                "choices": row.get("choices") or [],
                "precision": row.get("precision"),
            },
        })

    examples = _select_subset(examples, max_samples, seed)
    print(f"[dataset] MathVista: {len(examples)} examples")
    return examples


def load_mathvision(
    dataset_id: str,
    split: str,
    max_samples: Optional[int],
    seed: int,
) -> List[Dict]:
    """
    Load MathVision (competition math with images).

    Returns dicts with keys: id, image, question, answer, metadata
    """
    from datasets import load_dataset

    print(f"[dataset] Loading MathVision ({split})...")
    ds = load_dataset(dataset_id, split=split, trust_remote_code=True)

    examples = []
    for i, row in enumerate(ds):
        image = _to_pil(
            row.get("decoded_image") or row.get("image") or row.get("figure")
        )
        if image is None:
            continue

        question = str(row.get("question", row.get("problem", ""))).strip()
        answer = str(row.get("answer", row.get("solution", ""))).strip()

        # For multiple-choice rows, append choices
        choices = row.get("choices") or row.get("options")
        if choices and isinstance(choices, list):
            lettered = [f"({chr(65+j)}) {c}" for j, c in enumerate(choices)]
            question = question + "\nChoices: " + "  ".join(lettered)

        if not question or not answer:
            continue

        examples.append({
            "id": str(row.get("id", row.get("pid", i))),
            "image": image,
            "question": question,
            "answer": answer,
            "metadata": {
                "subject": row.get("subject", ""),
                "level": row.get("level", ""),
                "question_type": row.get("question_type", ""),
            },
        })

    examples = _select_subset(examples, max_samples, seed)
    print(f"[dataset] MathVision: {len(examples)} examples")
    return examples


def load_mmmu(
    dataset_id: str,
    split: str,
    max_samples: Optional[int],
    seed: int,
    subjects: Optional[list] = None,
) -> List[Dict]:
    """
    Load MMMU validation set.

    MMMU is split by subject on HuggingFace; we load multiple subjects and
    combine them, then sample to max_samples.

    Returns dicts with keys: id, image, question, answer, metadata
    """
    from datasets import load_dataset, get_dataset_config_names

    print(f"[dataset] Loading MMMU ({split})...")

    # Discover subject configs
    try:
        all_subjects = get_dataset_config_names(dataset_id)
    except Exception:
        # Fallback list covering the standard MMMU subjects
        all_subjects = [
            "Accounting", "Agriculture", "Architecture_and_Engineering",
            "Art", "Art_Theory", "Basic_Medical_Science", "Biology",
            "Chemistry", "Clinical_Medicine", "Computer_Science",
            "Design", "Diagnostics_and_Laboratory_Medicine", "Economics",
            "Electronics", "Energy_and_Power", "Finance", "Geography",
            "History", "Literature", "Manage", "Marketing",
            "Materials", "Math", "Mechanical_Engineering", "Music",
            "Pharmacy", "Physics", "Psychology", "Public_Health",
            "Sociology",
        ]

    if subjects:
        all_subjects = [s for s in all_subjects if s in subjects]

    examples = []
    for subj in all_subjects:
        try:
            ds = load_dataset(dataset_id, subj, split=split, trust_remote_code=True)
        except Exception as e:
            print(f"[dataset] MMMU: skipping {subj}: {e}")
            continue

        for row in ds:
            # MMMU can have up to 7 images; take the first non-None one
            image = None
            for img_key in ["image_1", "image_2", "image_3", "image_4"]:
                image = _to_pil(row.get(img_key))
                if image is not None:
                    break
            if image is None:
                # Some MMMU questions have no image; skip for VLM eval
                continue

            question = str(row.get("question", "")).strip()
            options_raw = row.get("options", [])
            if isinstance(options_raw, list) and options_raw:
                lettered = [f"({chr(65+j)}) {c}" for j, c in enumerate(options_raw)]
                question = question + "\n" + "  ".join(lettered)

            answer = str(row.get("answer", "")).strip()
            if not question or not answer:
                continue

            examples.append({
                "id": str(row.get("id", len(examples))),
                "image": image,
                "question": question,
                "answer": answer,
                "metadata": {
                    "subject": subj,
                    "topic": row.get("topic", ""),
                    "image_type": row.get("img_type", ""),
                },
            })

    examples = _select_subset(examples, max_samples, seed)
    print(f"[dataset] MMMU: {len(examples)} examples across {len(all_subjects)} subjects")
    return examples
