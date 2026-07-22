import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python generate_config.py <dataset_root> [model_name]")
    print("Example: python generate_config.py ./fleurs-shona-omni omniASR_CTC_300M")
    sys.exit(1)

root = Path(sys.argv[1])
model_name = sys.argv[2] if len(sys.argv) > 2 else "omniASR_CTC_300M"

config_name = f"{root.name}-{model_name.lower()}-finetune.yaml"

output_dir = Path.cwd() / "finetuning_output"
asr_dir = Path.cwd() / "omnilingual-asr"

print(f"""
Root: {root}
Model: {model_name}
Config: {config_name}
""")

config_path = Path(
    "omnilingual-asr/workflows/recipes/wav2vec2/asr/configs/"
    f"{root.name}-{model_name.lower()}-finetune.yaml"
)

config = f"""
model:
  name: "{model_name}"

dataset:
  name: "example_dataset"

  train_split: "train"
  valid_split: "dev"

  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"

  config_overrides:
    data: "{root}"

  mixture_parquet_storage_config:
    dataset_summary_path: "{root}/language_distribution_0.tsv"

    beta_corpus: 0.5
    beta_language: 0.5

    fragment_loading:
      cache: True

  asr_task_config:
    min_audio_len: 16000
    max_audio_len: 32000
    max_num_elements: 32000

    batch_shuffle_window: 1
    example_shuffle_window: 1
    normalize_audio: true

tokenizer:
  name: "omniASR_tokenizer_v1"

optimizer:
  config:
    lr: 5e-05

trainer:
  freeze_encoder_for_n_steps: 0

  mixed_precision:
    dtype: "torch.float16"

  grad_accumulation:
    num_batches: 1

regime:
  num_steps: 10

  validate_after_n_steps: 5
  validate_every_n_steps: 5

  checkpoint_every_n_steps: 5

  publish_metrics_every_n_steps: 5
"""

config_path.write_text(config)

print(f"Created config: {config_path}")
