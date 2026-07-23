import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python generate_config.py <dataset_root> [model_name] [--test]")
    print("Example: python generate_config.py ./fleurs-shona-omni omniASR_CTC_300M")
    print("Example (smoke test): python generate_config.py ./fleurs-shona-omni omniASR_CTC_300M --test")
    sys.exit(1)

root = Path(sys.argv[1])
model_name = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "omniASR_CTC_300M"
test_mode = "--test" in sys.argv

# Use absolute path or relative to current directory
if not root.is_absolute():
    root = Path.cwd() / root


def _count_train_groups(dataset_root: Path) -> tuple[int, int]:
    """Count unique corpora/languages available in the train split."""
    train_dirs = list(dataset_root.glob("corpus=*/split=train/language=*"))

    corpora = set()
    languages = set()

    for train_dir in train_dirs:
        # .../corpus=<name>/split=train/language=<name>
        corpus_part = train_dir.parents[1].name
        language_part = train_dir.name

        if corpus_part.startswith("corpus="):
            corpora.add(corpus_part.replace("corpus=", "", 1))

        if language_part.startswith("language="):
            languages.add(language_part.replace("language=", "", 1))

    return len(corpora), len(languages)


def _detect_valid_split(dataset_root: Path) -> str:
    """Pick validation split expected by the recipe from dataset folders."""
    split_names = {
        p.parents[0].name.replace("split=", "", 1)
        for p in dataset_root.glob("corpus=*/split=*/language=*")
        if p.parents[0].name.startswith("split=")
    }

    for candidate in ("dev", "validation", "valid"):
        if candidate in split_names:
            return candidate

    return "dev"


num_corpora, num_languages = _count_train_groups(root)
beta_corpus = 0.5 if num_corpora > 1 else 0.0
beta_language = 0.5 if num_languages > 1 else 0.0
valid_split = _detect_valid_split(root)

# Determine config output directory
# Look for omnilingual-asr in current directory or parent
script_dir = Path.cwd()
omni_dir = script_dir / "omnilingual-asr"

if not omni_dir.exists():
    # Try parent directory
    omni_dir = script_dir.parent / "omnilingual-asr"

config_dir = omni_dir / "workflows/recipes/wav2vec2/asr/configs"
config_dir.mkdir(parents=True, exist_ok=True)

config_name = f"{root.name}-{model_name.lower()}-finetune.yaml"
if test_mode:
    config_name = config_name.replace(".yaml", "-test.yaml")
config_path = config_dir / config_name

# Smoke test settings (minimal)
if test_mode:
    num_steps = 200
    validate_after = 50
    validate_every = 50
    checkpoint_every = 50
    publish_every = 10
    min_audio = 32_000
    max_audio = 960_000      
    max_num_elements = 960_000 
    grad_accum = 1          
    print("*** SMOKE TEST MODE ***")
else:
    # Model-specific settings
    is_llm = "llm" in model_name.lower()

    if is_llm:
        num_steps = 7_500
        validate_after = 0
        validate_every = 2_500
        checkpoint_every = 2_500
        publish_every = 2_500
        min_audio = 32_000
        max_audio = 160_000          # ← reduced from 320_000
        max_num_elements = 160_000   # ← reduced from 320_000
        grad_accum = 4               # ← increased from 1
    else:  # CTC model
        num_steps = 200
        validate_after = 50
        validate_every = 50
        checkpoint_every = 50
        publish_every = 1
        min_audio = 32_000
        max_audio = 960_000
        max_num_elements = 960_000
        grad_accum = 4

print(f"""
Root: {root}
Model: {model_name}
Config: {config_name}
Config dir: {config_dir}
Mode: {'TEST' if test_mode else 'PRODUCTION'}
Model type: {'LLM (FSDP)' if not test_mode and "llm" in model_name.lower() else 'CTC'}
Steps: {num_steps}
Max audio: {max_audio}
Grad accum: {grad_accum}
Train corpora: {num_corpora} (beta_corpus={beta_corpus})
Train languages: {num_languages} (beta_language={beta_language})
Validation split: {valid_split}
""")

config = f"""
model:
  name: "{model_name}"

dataset:
  name: "example_dataset"

  train_split: "train"
  valid_split: "{valid_split}"

  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"

  config_overrides:
    data: "{root}"

  mixture_parquet_storage_config:
    dataset_summary_path: "{root}/language_distribution_0.tsv"

    beta_corpus: {beta_corpus}
    beta_language: {beta_language}

    fragment_loading:
      cache: True

  asr_task_config:
    min_audio_len: {min_audio}
    max_audio_len: {max_audio}
    max_num_elements: {max_num_elements}

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
    dtype: "torch.bfloat16"

  grad_accumulation:
    num_batches: {grad_accum}
""" + ("""
  data_parallelism: "fsdp"
  fsdp:
    granularity: "stack"
    version: "v1"
    fp32_reduce: false
""" if not test_mode and "llm" in model_name.lower() else "") + f"""
regime:
  num_steps: {num_steps}

  validate_after_n_steps: {validate_after}
  validate_every_n_steps: {validate_every}

  checkpoint_every_n_steps: {checkpoint_every}

  publish_metrics_every_n_steps: {publish_every}
"""

config_path.write_text(config)

print(f"Created config: {config_path}")
