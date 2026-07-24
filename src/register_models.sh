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
name: lingala-omni-300M-5000
model_family: wav2vec2_asr
model_arch: 300m_v2
checkpoint: https://huggingface.co/KevinKibe/lingala-omni-300M/resolve/main/checkpoints/step_5000/model/pp_00/tp_00/sdp_00.pt
tokenizer_ref: omniASR_tokenizer_v1
EOF
re
read -r -d '' MODEL_2 << 'EOF' || true
---
name: lingala-omni-300M-10000
model_family: wav2vec2_asr
model_arch: 300m_v2
checkpoint: https://huggingface.co/KevinKibe/lingala-omni-300M/resolve/main/checkpoints/step_10000/model/pp_00/tp_00/sdp_00.pt
tokenizer_ref: omniASR_tokenizer_v1
EOF

echo ""
echo "==> Upserting custom model entries..."

# Remove any old/stale definitions (both list-item style and document style), then append canonical blocks.
python3 - "$MODELS_YAML" << 'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

names = [
    "omniASR_CTC_300M_lin_10k_test",
    "omniASR_CTC_300M_lin_best_test",
]

for name in names:
    # Remove malformed list-style entries:
    # - name: ...
    #   field: ...
    text = re.sub(
        rf"(?ms)^\s*-\s*name:\s*{re.escape(name)}\s*\n(?:^[ \t].*\n?)*",
        "",
        text,
    )

    # Remove proper/legacy multi-doc entries:
    # ---\nname: ...\n...
    text = re.sub(
        rf"(?ms)\n?^---\s*\nname:\s*{re.escape(name)}\s*\n(?:^(?!\s*---\s*$).*(?:\n|$))*",
        "",
        text,
    )

    # Remove possible no-separator doc entries:
    # name: ...\n...
    text = re.sub(
        rf"(?ms)^name:\s*{re.escape(name)}\s*\n(?:^(?!\s*---\s*$).*(?:\n|$))*",
        "",
        text,
    )

path.write_text(text.rstrip() + "\n", encoding="utf-8")
PY

printf "\n%s\n" "$MODEL_1" >> "$MODELS_YAML"
printf "\n%s\n" "$MODEL_2" >> "$MODELS_YAML"

echo "✓ Upserted omniASR_CTC_300M_lin_10k_test"
echo "✓ Upserted omniASR_CTC_300M_lin_best_test"

echo ""
echo "==> Registration complete!"
echo "Updated: $MODELS_YAML"
echo ""
echo "Registered models:"
grep "name: omniASR_CTC_300M_lin" "$MODELS_YAML" | sed 's/^/  /'
