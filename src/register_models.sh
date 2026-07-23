#!/bin/bash
set -e

# Find omnilingual-asr directory
echo "==> Looking for omnilingual-asr directory..."

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OMNI_DIR="$SCRIPT_DIR/omnilingual-asr"

if [ ! -d "$OMNI_DIR" ]; then
    # This shouldn't happen if setup.sh was run, but just in case
    echo "ERROR: omnilingual-asr directory not found at $OMNI_DIR"
    exit 1
fi

echo "Found omnilingual-asr at: $OMNI_DIR"

# Find rc_models_v1.yaml
MODELS_YAML="$OMNI_DIR/src/omnilingual_asr/cards/models/rc_models_v1.yaml"

if [ ! -f "$MODELS_YAML" ]; then
    echo "ERROR: rc_models_v1.yaml not found at: $MODELS_YAML"
    exit 1
fi

echo "Found models file at: $MODELS_YAML"

# Models to add
read -r -d '' MODEL_1 << 'EOF' || true
---
name: omniASR_CTC_300M_lin_10k_test
model_family: wav2vec2_asr
model_arch: 300m
checkpoint: https://huggingface.co/KevinKibe/omniASR-lingala-10k/resolve/main/model/pp_00/tp_00/sdp_00.pt
tokenizer_ref: omniASR_tokenizer_v1
EOF

read -r -d '' MODEL_2 << 'EOF' || true
---
name: omniASR_CTC_300M_lin_best_test
model_family: wav2vec2_asr
model_arch: 300m
checkpoint: https://huggingface.co/KevinKibe/omniASR-lingala-finetuned/resolve/main/model/pp_00/tp_00/sdp_00.pt
tokenizer_ref: omniASR_tokenizer_v1
EOF

# Check if models already exist
echo ""
echo "==> Checking if models already registered..."

if grep -q "omniASR_CTC_300M_lin_10k_test" "$MODELS_YAML"; then
    echo "✓ omniASR_CTC_300M_lin_10k_test already exists"
else
    echo "Adding omniASR_CTC_300M_lin_10k_test..."
    # Ensure we append on a fresh line to avoid corrupting YAML when file lacks trailing newline.
    printf "\n%s\n" "$MODEL_1" >> "$MODELS_YAML"
    echo "✓ Added omniASR_CTC_300M_lin_10k_test"
fi

if grep -q "omniASR_CTC_300M_lin_best_test" "$MODELS_YAML"; then
    echo "✓ omniASR_CTC_300M_lin_best_test already exists"
else
    echo "Adding omniASR_CTC_300M_lin_best_test..."
    # Ensure we append on a fresh line to avoid corrupting YAML when file lacks trailing newline.
    printf "\n%s\n" "$MODEL_2" >> "$MODELS_YAML"
    echo "✓ Added omniASR_CTC_300M_lin_best_test"
fi

echo ""
echo "==> Registration complete!"
echo "Updated: $MODELS_YAML"
echo ""
echo "Registered models:"
grep "name: omniASR_CTC_300M_lin" "$MODELS_YAML" | sed 's/^/  /'
