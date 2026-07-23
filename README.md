# Omnilingual ASR Finetuning

Production pipeline for finetuning Meta's Omnilingual ASR models (CTC_300M, LLM_1B) on custom datasets.

## Quick Start

### Local
```bash
bash src/setup.sh  # One-time setup
bash src/finetune.sh <dataset_repo> <model_name> [--test]
```

### Docker
```bash
docker build -t asr-finetuning .
docker run --gpus all -it asr-finetuning:latest <dataset_repo> <model_name> [--test]
```

## Examples

```bash
# Smoke test (20 steps, ~5 min)
bash src/finetune.sh KevinKibe/fleurs-lingala-omni omniASR_CTC_300M --test

# Combined Shona training data: FLEURS train/dev/test + Waxal train/test.
# This only checks data composition; smoke tests skip validation.
bash src/finetune.sh KevinKibe/fleurs-shona-omni omniASR_CTC_300M \
  --combine-waxal-sna --test

# Full combined Shona run. Validation uses only Waxal's validation split.
bash src/finetune.sh KevinKibe/fleurs-shona-omni omniASR_CTC_300M \
  --combine-waxal-sna

# Full combined Lingala run: FLEURS train/dev/test + Waxal train/test.
# Validation uses only Waxal's validation split.
bash src/finetune.sh KevinKibe/fleurs-lingala-omni omniASR_CTC_300M \
  --combine-waxal-lin

# Full combined Luganda run: FLEURS train/dev/test + Waxal train/test.
# Validation uses only Waxal's validation split.
bash src/finetune.sh KevinKibe/fleurs-luganda-omni omniASR_CTC_1B_v2 --combine-waxal-lug

# Full training (10k steps)
bash src/finetune.sh KevinKibe/fleurs-lingala-omni omniASR_CTC_300M

# Full training with upload to HuggingFace
bash src/finetune.sh KevinKibe/fleurs-lingala-omni omniASR_CTC_300M \
  KevinKibe/omniASR-lingala-finetuned hf_xxxxx

# Combined Lingala LLM run with FSDP
bash src/finetune.sh KevinKibe/fleurs-lingala-omni omniASR_LLM_1B \
  --combine-waxal-lin
```


## Docker GPU Selection

```bash
docker run --gpus all ...       # all GPUs
docker run --gpus 0 ...         # GPU 0 only
docker run --gpus 0,1,2 ...     # specific GPUs
```


## Inference On WaxalNLP

Run your finetuned model card against WaxalNLP test samples for any ASR language.

```bash
# Example: Luganda test inference with a custom registered card.
# --omni-lang is recommended for LLM models and optional for CTC.
python src/infer_waxalnlp.py \
  --model-card omniASR_CTC_300M_lin_best_test \
  --language lug \
  --omni-lang lug_Latn \
  --split test \
  --batch-size 4

# Quick smoke inference on first 100 samples.
python src/infer_waxalnlp.py \
  --model-card omniASR_CTC_300M_lin_best_test \
  --language lin \
  --max-samples 100 \
  --streaming
```

Outputs are written under `outputs/inference/`:
- `*.jsonl`: per-sample predictions and references.
- `*.metrics.json`: summary metrics (WER/CER).

Notes:
- Waxal ASR config is inferred as `<language>_asr` (for example: `lug_asr`, `lin_asr`, `sna_asr`).
- Use `--config` to override config name manually.
- Omnilingual currently accepts audio samples shorter than 40 seconds for this inference path.



Built on [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) and [fairseq2](https://github.com/facebookresearch/fairseq2)
