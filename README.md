# Stage 4 — Reinforcement Learning Post-Training

Roadmap stage 4 (9 Nov – 22 Nov 2026).  
Model: **Qwen2.5-VL-7B-Instruct** · GPU: Colab L4 (22.5 GB)

## Experiments

| ID   | Script                   | Status | Description |
|------|--------------------------|--------|-------------|
| E4.1 | `E4_1_sft_baseline.py`   | ✅     | SFT on LLaVA-CoT-100k; eval on MathVista / MathVision / MMMU |
| E4.2 | `E4_2_grpo.py`           | —      | GRPO reproduction (VLM-R1 / EasyR1) |
| E4.3 | `E4_3_cold_start.py`     | —      | Ablation: cold-start CoT vs RL-only |
| E4.4 | `E4_4_reward_surgery.py` | —      | Novel reward component (grounding-consistency) |
| E4.5 | `E4_5_overthinking.py`   | —      | Accuracy vs reasoning-length audit |
| E4.6 | `E4_6_transfer.py`       | —      | Out-of-domain generalisation test |
| E4.7 | `E4_7_quality_rl.py`     | —      | MAVERIC-style quality-aware RL data selection |

## Setup

```bash
# 1. Install dependencies
pip install torch torchvision transformers accelerate datasets peft trl \
            pyyaml huggingface_hub wandb flash-attn pillow packaging

# 2. Validate environment
python setup.py

# 3. (Optional) login to HuggingFace for gated models
huggingface-cli login
```

## E4.1 — SFT Baseline

```bash
# Full run  (train on 50k samples, then eval)
python E4_1_sft_baseline.py

# Quick smoke-test (1k train, 100 eval each)
python E4_1_sft_baseline.py \
    --set sft.max_samples=1000 \
    --set eval.mathvista.max_samples=100 \
    --set eval.mathvision.max_samples=100 \
    --set eval.mmmu.max_samples=100

# Eval only from a saved checkpoint
python E4_1_sft_baseline.py \
    --mode eval \
    --checkpoint outputs/e4_1/<run_name>/adapter_final

# Re-score existing JSONL without GPU
python E4_1_sft_baseline.py \
    --mode analyze \
    --run-dir outputs/e4_1/<run_name>
```

Config overrides use dot-notation:
```bash
--set model.name=Qwen/Qwen2.5-VL-3B-Instruct  # smaller model
--set sft.per_device_train_batch_size=1         # if OOM
--set model.use_flash_attn=false                # if flash-attn not installed
```

## Repository layout

```
research_stage_04/
├── config.yaml                 # all experiment parameters
├── setup.py                    # environment check
├── E4_1_sft_baseline.py        # SFT + eval
├── common/
│   ├── config.py               # config loading, run dirs, seeds
│   ├── dataset_loaders.py      # SFT data + eval benchmarks
│   ├── scoring_logic.py        # answer extraction + accuracy
│   └── collator.py             # VLM data collator (label masking)
├── docs/                       # result write-ups
└── outputs/                    # git-ignored run artifacts
```

## Expected results (E4.1 reference)

| Benchmark   | Base model (0-shot) | After SFT |
|-------------|--------------------:|----------:|
| MathVista   | ~59%               | TBD       |
| MathVision  | ~18%               | TBD       |
| MMMU        | ~54%               | TBD       |

*Fill in after running E4.1. These numbers become the baseline for all E4.2+ RL claims.*
