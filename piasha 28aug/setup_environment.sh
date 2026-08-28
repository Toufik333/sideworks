#!/bin/bash
# ============================================================================
# SBERT Paper Reproduction — Apple Silicon (M1) Environment Setup
# Based on: Reimers & Gurevych (2019), "Sentence-BERT: Sentence Embeddings
#           using Siamese BERT-Networks" (EMNLP 2019)
# Repository: https://github.com/UKPLab/sentence-transformers
# ============================================================================

set -e

VENV_DIR="sbert_env"
PYTHON_CMD="python3"

echo "============================================================"
echo "  SBERT Paper Reproduction — Environment Setup"
echo "  Target: Apple Silicon (M1/M2/M3) with MPS acceleration"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------
# 1. Check Python version (>= 3.9 required for MPS support)
# --------------------------------------------------------------------------
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

echo "[1/5] Checking Python version..."
echo "       Found: Python $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "ERROR: Python >= 3.9 is required for MPS support."
    echo "       Install via: brew install python@3.11"
    exit 1
fi
echo "       ✓ Python version is compatible."
echo ""

# --------------------------------------------------------------------------
# 2. Create virtual environment
# --------------------------------------------------------------------------
echo "[2/5] Creating virtual environment: $VENV_DIR"

if [ -d "$VENV_DIR" ]; then
    echo "       Virtual environment already exists. Reusing."
else
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "       ✓ Virtual environment created."
fi
echo ""

# --------------------------------------------------------------------------
# 3. Activate and upgrade pip
# --------------------------------------------------------------------------
echo "[3/5] Activating environment and upgrading pip..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel --quiet
echo "       ✓ pip upgraded."
echo ""

# --------------------------------------------------------------------------
# 4. Install PyTorch with MPS support + sentence-transformers
# --------------------------------------------------------------------------
echo "[4/5] Installing dependencies..."
echo "       → PyTorch (MPS backend for Apple Silicon)"
echo "       → sentence-transformers (UKPLab/sentence-transformers)"
echo "       → datasets, scipy, scikit-learn, matplotlib"
echo ""

pip install --quiet \
    torch \
    torchvision \
    torchaudio \
    sentence-transformers \
    transformers \
    datasets \
    scipy \
    scikit-learn \
    matplotlib \
    tqdm \
    numpy

echo "       ✓ All packages installed."
echo ""

# --------------------------------------------------------------------------
# 5. Verify installation and device detection
# --------------------------------------------------------------------------
echo "[5/5] Verifying installation..."
echo ""

python3 -c "
import torch
import sentence_transformers
import transformers
import datasets

print('  Package Versions:')
print(f'    PyTorch:               {torch.__version__}')
print(f'    sentence-transformers: {sentence_transformers.__version__}')
print(f'    transformers:          {transformers.__version__}')
print(f'    datasets:              {datasets.__version__}')
print()

# Device detection: MPS → CPU fallback
if torch.backends.mps.is_available():
    device = torch.device('mps')
    device_name = 'Apple Silicon GPU (MPS)'
elif torch.cuda.is_available():
    device = torch.device('cuda')
    device_name = f'CUDA GPU ({torch.cuda.get_device_name(0)})'
else:
    device = torch.device('cpu')
    device_name = 'CPU'

print(f'  Compute Device: {device_name}')
print(f'  torch.device:   {device}')
print()

# Quick smoke test
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))
emb = model.encode(['Hello world'], convert_to_tensor=True)
print(f'  Smoke Test: ✓ Embedding shape = {emb.shape}')
print(f'  Embedding device: {emb.device}')
print()
print('  ✅ Environment setup complete!')
"

echo ""
echo "============================================================"
echo "  Setup complete! Activate the environment with:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Then run:"
echo "    python script_a_inference.py     # Inference & Evaluation"
echo "    python script_b_finetune.py      # Fine-tuning Demo"
echo "============================================================"
