#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to check if a command exists
command_exists () {
    type "$1" &> /dev/null ;
}

# Update pip
pip install --upgrade pip

export MASTER_ADDR=localhost
export MASTER_PORT=12355
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Detect CUDA version
if command_exists nvidia-smi && command_exists nvcc; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//') 
    CUDA_VERSION_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_VERSION_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    CUDA_VERSION_SHORT="${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}"
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "CUDA not detected. Installing CPU version of PyTorch."
    CUDA_VERSION_SHORT="cpu"
fi

# Install PyTorch
if [ "$CUDA_VERSION_SHORT" = "cpu" ]; then
    pip install torch torchvision torchaudio
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}
fi

# Get installed PyTorch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" | cut -d+ -f1)

# Install torch-scatter and torch-sparse
if [ "$CUDA_VERSION_SHORT" = "cpu" ]; then
    pip install torch-scatter torch-sparse
else
    pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION_SHORT}.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION_SHORT}.html
fi

# Install requirements (excluding torch and torch-sparse for now)
pip install pyarrow torch_geometric biopython dask rdkit pandas scikit-learn

# Run the training script
python training.py