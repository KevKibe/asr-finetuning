#!/usr/bin/env bash
set -euo pipefail

# Omnilingual ASR Finetuning Orchestrator
# Downloads dataset, generates config, runs training, and uploads model

usage() {
    cat << 'EOF'
Usage: ./finetune.sh <dataset_repo> <model_name> [output_repo] [hf_token] [--combine-waxal] [--test]

Arguments:
  dataset_repo       HuggingFace dataset repo ID (e.g., KevinKibe/fleurs-shona-omni)
  model_name        Model to finetune (e.g., omniASR_CTC_300M, omniASR_LLM_1B)
  output_repo       HuggingFace repo to upload model (default: <dataset_repo>-finetuned)
  hf_token          HuggingFace API token (default: $HF_TOKEN env var)
    --combine-waxal
                                        Combine supported FLEURS train/dev/test with matching Waxal
                                        train/test; use Waxal validation for evaluation
    --test            Run the smoke-test configuration (validation is skipped)

Example:
  ./finetune.sh KevinKibe/fleurs-shona-omni omniASR_CTC_300M KevinKibe/omniASR-fleurs-finetuned
    ./finetune.sh KevinKibe/fleurs-shona-omni omniASR_CTC_300M --combine-waxal
    ./finetune.sh KevinKibe/fleurs-lingala-omni omniASR_CTC_300M --combine-waxal
    ./finetune.sh KevinKibe/fleurs-luganda-omni omniASR_CTC_300M --combine-waxal

Environment:
  HF_TOKEN          Your HuggingFace API token (for upload)
EOF
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

DATASET_REPO="$1"
MODEL_NAME="$2"

# Parse optional flags
TEST_FLAG=""
COMBINE_WAXAL=false
OUTPUT_REPO="${DATASET_REPO}-finetuned"
HF_TOKEN="${HF_TOKEN:-}"

for arg in "${@:3}"; do
    case "$arg" in
        --test) TEST_FLAG="--test" ;;
        --combine-waxal|--combine-waxal-sna) COMBINE_WAXAL=true ;;
        hf_*|*:*) HF_TOKEN="$arg" ;;
        *) OUTPUT_REPO="$arg" ;;
    esac
done

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_step() {
    echo -e "${BLUE}==>${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
DATASET_NAME=$(basename "$DATASET_REPO")
DATASET_DIR="$SCRIPT_DIR/$DATASET_NAME"
OUTPUT_DIR="$SCRIPT_DIR/finetuning_output"
EXPORT_DIR="$SCRIPT_DIR/hf_export"

log_step "OmniASR Finetuning Workflow"
echo "Dataset repo: $DATASET_REPO"
echo "Model: $MODEL_NAME"
echo "Output repo: $OUTPUT_REPO"
echo "Dataset dir: $DATASET_DIR"
echo ""

# Step 1: Download dataset
log_step "Downloading dataset from HuggingFace..."
cd "$SCRIPT_DIR"
python3 "$SRC_DIR/dataset_download.py" "$DATASET_REPO" "$DATASET_DIR"
log_success "Dataset downloaded"
echo ""

if [[ "$COMBINE_WAXAL" == true ]]; then
    case "$DATASET_REPO" in
        KevinKibe/fleurs-shona-omni)
            WAXAL_REPO="KevinKibe/waxal-sna-omni"
            LANGUAGE_LABEL="Shona"
            ;;
        KevinKibe/fleurs-lingala-omni)
            WAXAL_REPO="KevinKibe/waxal-lin-omni"
            LANGUAGE_LABEL="Lingala"
            ;;
        KevinKibe/fleurs-luganda-omni)
            WAXAL_REPO="KevinKibe/waxal-lug-omni"
            LANGUAGE_LABEL="Luganda"
            ;;
        *)
            log_error "--combine-waxal supports FLEURS Shona, Lingala, or Luganda datasets"
            exit 1
            ;;
    esac

    WAXAL_DIR="$SCRIPT_DIR/$(basename "$WAXAL_REPO")"
    COMBINED_DATASET_DIR="$SCRIPT_DIR/${DATASET_NAME}-$(basename "$WAXAL_REPO")-combined"

    log_step "Downloading Waxal $LANGUAGE_LABEL dataset from HuggingFace..."
    python3 "$SRC_DIR/dataset_download.py" "$WAXAL_REPO" "$WAXAL_DIR"
    log_success "Waxal dataset downloaded"

    log_step "Building combined FLEURS and Waxal $LANGUAGE_LABEL dataset..."
    python3 "$SRC_DIR/prepare_combined_shona_dataset.py" \
        "$DATASET_DIR" "$WAXAL_DIR" "$COMBINED_DATASET_DIR"
    DATASET_DIR="$COMBINED_DATASET_DIR"
    DATASET_NAME=$(basename "$DATASET_DIR")
    log_success "Combined dataset built"
    echo ""
fi

# Step 2: Generate language distribution
log_step "Generating language distribution..."
python3 "$SRC_DIR/lang_distribution.py" "$DATASET_DIR"
log_success "Language distribution generated"
echo ""

# Step 3: Generate config
log_step "Generating finetuning config..."
python3 "$SRC_DIR/generate_config.py" "$DATASET_DIR" "$MODEL_NAME" ${TEST_FLAG}
log_success "Config generated"
echo ""

# Step 4: Run finetuning
BASE_CONFIG="$DATASET_NAME-$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')-finetune.yaml"
CONFIG_NAME="${BASE_CONFIG%.yaml}${TEST_FLAG:+-test}.yaml"

log_step "Starting finetuning..."
echo "Config: $CONFIG_NAME"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$SCRIPT_DIR"
mkdir -p "$OUTPUT_DIR"

# Ensure omnilingual-asr exists
ASR_DIR="$SCRIPT_DIR/omnilingual-asr"
if [ ! -d "$ASR_DIR" ]; then
    log_error "omnilingual-asr directory not found at $ASR_DIR"
    log_error "Run setup.sh first."
    exit 1
fi

export PYTHONPATH="$ASR_DIR/src:$ASR_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python3 -u -m workflows.recipes.wav2vec2.asr \
    "$OUTPUT_DIR" \
    --config-file "$ASR_DIR/workflows/recipes/wav2vec2/asr/configs/$CONFIG_NAME" \
    --config dataset.config_overrides.data="$DATASET_DIR" \
    2>&1 | tee "$SCRIPT_DIR/training.log"

log_success "Finetuning completed"

# Step 5: Extract best checkpoint
log_step "Extracting best checkpoint..."
python3 "$SRC_DIR/get_best_checkpoint.py" "$OUTPUT_DIR" "$EXPORT_DIR"
log_success "Best checkpoint extracted"
echo ""

# Step 6: Upload model (if token provided)
if [[ -n "$HF_TOKEN" ]]; then
    log_step "Uploading model to HuggingFace..."
    python3 "$SRC_DIR/upload_model.py" "$OUTPUT_REPO" "$EXPORT_DIR" "$HF_TOKEN"
    log_success "Model uploaded"
else
    echo "HF_TOKEN not provided, skipping upload"
    echo "To upload, run:"
    echo "  export HF_TOKEN=your_token_here"
    echo "  python3 $SRC_DIR/upload_model.py \"$OUTPUT_REPO\" \"$EXPORT_DIR\""
fi

echo ""
log_success "Workflow complete!"
echo "Results saved to: $EXPORT_DIR"
