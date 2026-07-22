#!/usr/bin/env bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up Omnilingual ASR environment...${NC}"

# Clone repository
echo -e "${YELLOW}Cloning omnilingual-asr repository...${NC}"
if [ -d "omnilingual-asr" ]; then
    echo "omnilingual-asr already exists, skipping clone"
else
    git clone https://github.com/facebookresearch/omnilingual-asr
fi
cd omnilingual-asr

# Install base dependencies
echo -e "${YELLOW}Installing base package dependencies...${NC}"
uv pip install -e .

# Uninstall conflicting versions
echo -e "${YELLOW}Removing conflicting torchaudio and torchvision versions...${NC}"
uv pip uninstall -y torchaudio torchvision || true

# Install CUDA-optimized PyTorch packages
echo -e "${YELLOW}Installing PyTorch packages with CUDA support...${NC}"
uv pip install --no-cache-dir \
    torchaudio==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

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
