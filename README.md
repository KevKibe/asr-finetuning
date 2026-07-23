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
bash src/finetune.sh KevinKibe/fleurs-luganda-omni omniASR_CTC_300M \
  --combine-waxal-lug

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



Built on [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) and [fairseq2](https://github.com/facebookresearch/fairseq2)
