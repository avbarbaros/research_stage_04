"""
common/scoring_logic.py — Answer extraction and correctness scoring for eval benchmarks.

Each scorer takes (response: str, gt_answer: str, metadata: dict) → bool.
"""

from __future__ import annotations

import math
import re
import string
from typing import Any, Dict, List, Optional, Tuple


# ── Low-level extraction ──────────────────────────────────────────────────────

def _extract_boxed(text: str) -> Optional[str]:
    """Extract the innermost \\boxed{...} content."""
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def _extract_last_number(text: str) -> Optional[str]:
    """Find the last standalone number (int or float) in a string."""
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    return matches[-1] if matches else None


def _answer_region(text: str) -> str:
    """
    The part of a response that holds the final answer.

    LLaVA-CoT-style models answer as <SUMMARY>…<CAPTION>…<REASONING>…<CONCLUSION>…;
    the reasoning mentions many letters/numbers, so score only the conclusion
    when present (also handles an unterminated conclusion cut off by max_new_tokens).
    """
    m = re.findall(r"<CONCLUSION>(.*?)(?:</CONCLUSION>|$)", text, re.S | re.I)
    if m and m[-1].strip():
        return m[-1].strip()
    return text


def _extract_choice_letter(text: str, choices: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract the multiple-choice answer letter (A-J) from a response.

    Uses the LAST match of the strongest pattern found (CoT text discusses
    several options before committing). Letters must be uppercase, so the
    article "a" / pronoun "I" in prose are not mistaken for answers.
    Falls back to matching the option text itself when `choices` is given.
    """
    text = _answer_region(text).strip()
    if not text:
        return None

    patterns = [
        r"(?i:answer|choice|option)(?:\s+is)?\s*[:：]?\s*\**\s*\(?([A-J])\)?(?![A-Za-z])",
        r"\\boxed\{\s*\(?([A-J])\)?\s*\}",
        r"\*\*\s*\(?([A-J])\)?\s*\*\*",                 # **B** / **(B)**
        r"(?<![A-Za-z])\(([A-J])\)(?![A-Za-z])",            # (B)
        r"^\s*([A-J])\s*(?:[.:)]|$)",                       # starts with "B." / just "B"
        r"(?<![A-Za-z])([A-J])\s*[.)]?\s*$",                 # ends with "B" / "B."
    ]
    for pat in patterns:
        m = re.findall(pat, text)
        if m:
            return m[-1].upper()

    # Fallback: the response states the option text instead of its letter
    if choices:
        low = text.lower()
        hits = [(low.rfind(str(c).strip().lower()), i) for i, c in enumerate(choices)
                if str(c).strip() and str(c).strip().lower() in low]
        if hits:
            return chr(65 + max(hits)[1])
    return None


def _normalize_number(s: str) -> Optional[float]:
    """Try to parse a string as a float; return None on failure."""
    try:
        s = s.replace(",", "").strip()
        return float(s)
    except (ValueError, TypeError):
        return None


# ── MathVista scorer ──────────────────────────────────────────────────────────

def score_mathvista(response: str, gt_answer: str, metadata: Dict[str, Any]) -> bool:
    """
    Score a MathVista response.

    MathVista has two question_type families:
    - multi_choice  → compare extracted letter to ground truth letter
    - free_form     → numeric comparison (with precision tolerance)

    The logic follows the official MathVista evaluation script.
    """
    q_type = metadata.get("question_type", "")
    a_type = metadata.get("answer_type", "")
    response = _answer_region(response or "")

    if q_type == "multi_choice" or a_type in ("text", ""):
        # Multiple-choice
        choices = metadata.get("choices", [])
        pred_letter = _extract_choice_letter(response, choices)
        gt_letter = gt_answer.strip().upper()

        # Sometimes gt_answer is the full option text rather than a letter
        if len(gt_letter) > 1 and choices:
            # Find which choice matches gt_answer text
            for idx, ch in enumerate(choices):
                if ch.strip().lower() == gt_answer.strip().lower():
                    gt_letter = chr(65 + idx)
                    break

        return pred_letter is not None and pred_letter == gt_letter

    else:
        # Free-form / integer / float
        pred_text = (
            _extract_boxed(response)
            or _extract_last_number(response)
            or response.split()[-1] if response else ""
        )
        pred_num = _normalize_number(str(pred_text))
        gt_num = _normalize_number(gt_answer)

        if pred_num is None or gt_num is None:
            return str(pred_text).strip().lower() == gt_answer.strip().lower()

        # Use precision from metadata if available
        precision = metadata.get("precision")
        if precision is not None:
            tol = 10 ** (-precision)
            return abs(pred_num - gt_num) <= tol
        else:
            # Relative tolerance for large numbers, absolute for small
            if gt_num == 0:
                return abs(pred_num - gt_num) < 1e-6
            return abs(pred_num - gt_num) / (abs(gt_num) + 1e-9) < 0.01


# ── MathVision scorer ─────────────────────────────────────────────────────────

def score_mathvision(response: str, gt_answer: str, metadata: Dict[str, Any]) -> bool:
    """
    Score a MathVision response.

    MathVision is competition math; answers are almost always numeric or simple
    expressions. We extract the last \\boxed{} or last number and compare.
    """
    response = _answer_region(response or "")
    pred = _extract_boxed(response) or _extract_last_number(response)
    if pred is None:
        pred = response.strip().split()[-1] if response.strip() else ""

    pred_num = _normalize_number(pred)
    gt_num = _normalize_number(gt_answer)

    if pred_num is not None and gt_num is not None:
        if gt_num == 0:
            return abs(pred_num) < 1e-6
        return abs(pred_num - gt_num) / (abs(gt_num) + 1e-9) < 0.01

    # String fallback
    return str(pred).strip().lower() == gt_answer.strip().lower()


# ── MMMU scorer ───────────────────────────────────────────────────────────────

def score_mmmu(response: str, gt_answer: str, metadata: Dict[str, Any]) -> bool:
    """
    Score an MMMU response.

    Most MMMU items are multiple-choice (ground truth = one letter like 'A');
    ~5% are open-ended (question_type == "open") and are scored like
    MathVision (numeric tolerance, else exact string match).
    """
    gt = gt_answer.strip()
    if metadata.get("question_type") == "open" or not (
        len(gt) == 1 and gt.upper() in string.ascii_uppercase
    ):
        return score_mathvision(response, gt, metadata)

    pred_letter = _extract_choice_letter(response, metadata.get("choices"))
    return pred_letter == gt.upper()


# ── Aggregate statistics ──────────────────────────────────────────────────────

def wilson_ci(correct: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def accuracy_report(label: str, correct: int, total: int) -> str:
    if total == 0:
        return f"{label}: N/A (0 items)"
    acc = correct / total
    lo, hi = wilson_ci(correct, total)
    return f"{label}: {acc:.4f}  ({correct}/{total})  95% CI [{lo:.4f}, {hi:.4f}]"
