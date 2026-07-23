#!/usr/bin/env bash
set -euo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for the rest of this script
export PATH="$HOME/.local/bin:$PATH"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo -e "${YELLOW}Setting up Omnilingual ASR environment...${NC}"
echo "Script directory: $SCRIPT_DIR"

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
apt-get update -qq && apt-get install -y --no-install-recommends libsndfile1 ffmpeg git

# Clone repository into script directory
echo -e "${YELLOW}Cloning omnilingual-asr repository...${NC}"
if [ -d "$SCRIPT_DIR/omnilingual-asr" ]; then
    echo "omnilingual-asr already exists, skipping clone"
else
    cd "$SCRIPT_DIR"
    git clone https://github.com/facebookresearch/omnilingual-asr
fi
cd "$SCRIPT_DIR/omnilingual-asr"

# Install base dependencies
echo -e "${YELLOW}Installing base package dependencies...${NC}"
uv pip install --system -e .

# Uninstall conflicting versions
echo -e "${YELLOW}Removing conflicting torchaudio and torchvision versions...${NC}"
uv pip uninstall --system -y torchaudio torchvision || true

# Install CUDA-optimized PyTorch packages
echo -e "${YELLOW}Installing PyTorch packages with CUDA support...${NC}"
uv pip install --system --no-cache-dir \
    torchaudio==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install tensorboard
echo -e "${YELLOW}Installing tensorboard...${NC}"
uv pip install --system tensorboard

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
python3 << 'EOF'
import torch
import torchaudio

print(f"✓ torch: {torch.__version__}")
print(f"✓ cuda: {torch.version.cuda}")
print(f"✓ torchaudio: {torchaudio.__version__}")
EOF

echo -e "${GREEN}Setup completed successfully!${NC}"
