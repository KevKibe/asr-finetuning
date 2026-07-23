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

# Full training (10k steps)
bash src/finetune.sh KevinKibe/fleurs-lingala-omni omniASR_CTC_300M

# Full training with upload to HuggingFace
bash src/finetune.sh KevinKibe/fleurs-lingala-omni omniASR_CTC_300M \
  KevinKibe/omniASR-lingala-finetuned hf_xxxxx

# LLM model with FSDP
bash src/finetune.sh KevinKibe/fleurs-lingala-omni omniASR_LLM_1B
```


## Docker GPU Selection

```bash
docker run --gpus all ...       # all GPUs
docker run --gpus 0 ...         # GPU 0 only
docker run --gpus 0,1,2 ...     # specific GPUs
```



Built on [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) and [fairseq2](https://github.com/facebookresearch/fairseq2)
