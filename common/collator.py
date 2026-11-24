"""
common/collator.py — Data collator for SFT with Qwen2.5-VL.

Handles:
  • Formatting messages via the processor's chat template
  • Label masking: prompt tokens → -100, only assistant response computes loss
  • Manual padding + stacking of variable-length sequences with image tensors
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from PIL import Image


class VLMSFTCollator:
    """
    Collates SFT examples into batched model inputs for Qwen2.5-VL.

    Each example must have keys: image (PIL), question (str), answer (str).

    Label masking strategy
    ----------------------
    For each example we:
      1. Encode the FULL conversation (user + assistant).
      2. Encode the PROMPT ONLY (user turn with add_generation_prompt=True)
         using the same image so visual token counts match exactly.
      3. Mask labels[0 : prompt_length] = -100.
      4. Also mask padding positions.

    This guarantees the loss is computed only on the assistant's response tokens.
    """

    def __init__(self, processor: Any, max_seq_len: int = 2048):
        self.processor = processor
        self.max_seq_len = max_seq_len

        # Determine pad token id
        tok = processor.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # ── Public interface ──────────────────────────────────────────────────────

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        processed = [self._process_one(ex) for ex in batch]
        return self._collate(processed)

    # ── Single-example processing ─────────────────────────────────────────────

    def _process_one(self, ex: Dict) -> Dict[str, Any]:
        image: Image.Image = ex["image"]
        question: str = ex["question"]
        answer: str = ex["answer"]

        # 1. Build full-conversation text (user + assistant)
        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # 2. Build prompt-only text (user turn only, with generation prompt)
        prompt_messages = [full_messages[0]]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # 3. Encode the full conversation
        full_enc = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        )

        # 4. Encode prompt only (no truncation needed — it's always shorter)
        prompt_enc = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]
        # Guard: prompt_len should not exceed the (possibly truncated) full length
        prompt_len = min(prompt_len, full_enc["input_ids"].shape[1])

        # 5. Build labels: mask prompt tokens and padding
        labels = full_enc["input_ids"][0].clone()
        labels[:prompt_len] = -100
        # Mask padding (attention_mask == 0)
        labels[full_enc["attention_mask"][0] == 0] = -100

        return {
            "input_ids": full_enc["input_ids"][0],               # (seq_len,)
            "attention_mask": full_enc["attention_mask"][0],      # (seq_len,)
            "pixel_values": full_enc.get("pixel_values"),         # (patches, C, H, W) or None
            "image_grid_thw": full_enc.get("image_grid_thw"),     # (1, 3) or None
            "labels": labels,                                      # (seq_len,)
        }

    # ── Batching ──────────────────────────────────────────────────────────────

    def _collate(self, items: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(it["input_ids"].shape[0] for it in items)

        input_ids_list, attn_list, label_list = [], [], []
        pv_list, thw_list = [], []

        for it in items:
            seq_len = it["input_ids"].shape[0]
            pad_len = max_len - seq_len

            input_ids_list.append(
                torch.cat([it["input_ids"],
                           torch.full((pad_len,), self.pad_id, dtype=torch.long)])
            )
            attn_list.append(
                torch.cat([it["attention_mask"],
                           torch.zeros(pad_len, dtype=torch.long)])
            )
            label_list.append(
                torch.cat([it["labels"],
                           torch.full((pad_len,), -100, dtype=torch.long)])
            )

            if it["pixel_values"] is not None:
                pv_list.append(it["pixel_values"])
            if it["image_grid_thw"] is not None:
                thw_list.append(it["image_grid_thw"])

        result: Dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids_list),      # (B, max_len)
            "attention_mask": torch.stack(attn_list),       # (B, max_len)
            "labels": torch.stack(label_list),              # (B, max_len)
        }

        if pv_list:
            # pixel_values: each is (patches_i, C, H, W) → cat on dim 0
            result["pixel_values"] = torch.cat(pv_list, dim=0)
        if thw_list:
            # image_grid_thw: each is (1, 3) → cat on dim 0 → (B, 3)
            result["image_grid_thw"] = torch.cat(thw_list, dim=0)

        return result
