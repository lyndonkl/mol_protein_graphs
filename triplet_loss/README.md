# Triplet Loss Model Training

This repository contains code for training a graph neural network model using triplet loss for molecular similarity learning. The model is designed to learn molecular representations by comparing binding and non-binding molecule pairs.

## Project Structure

```
triplet_loss/
├── scripts/
│   ├── training_analysis.py    # Grid search and dataset size analysis
│   └── train.py               # Main training script
├── src/
│   ├── data/
│   │   ├── datasets.py        # Dataset implementations
│   │   ├── molecule.py        # Molecule data processing
│   │   └── preprocessor.py    # Data preprocessing utilities
│   ├── model/
│   │   ├── block.py          # Model building blocks
│   │   └── model.py          # Main model architecture
│   ├── training/
│   │   ├── evaluator.py      # Model evaluation
│   │   └── train.py          # Training loop implementation
│   └── utils/
│       └── helpers.py         # Utility functions
└── notebooks/
    └── Data Analysis.ipynb    # Data exploration and analysis
```

## Features

- Distributed training support using PyTorch DDP
- Grid search for hyperparameter optimization
- Molecule graph representation using PyTorch Geometric
- Triplet loss for learning molecular similarities
- Comprehensive evaluation metrics

## Model Architecture

The model uses a stacked architecture with:
- Graph neural network layers for molecular structure processing
- Self-attention mechanisms for feature learning
- Protein type embeddings
- Triplet margin loss for similarity learning

## Usage

1. Data Preprocessing:
```python
from triplet_loss.src.data.preprocessor import Preprocessor

preprocessor = Preprocessor("train.parquet", "output_dir")
preprocessor.process()
```

2. Training:
```python
python triplet_loss/scripts/train.py
```

3. Hyperparameter Optimization:
```python
python triplet_loss/scripts/training_analysis.py
```

## Requirements

- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- Pandas
- scikit-learn

## Training Parameters

The model supports various hyperparameters that can be tuned:

```python
param_grid = {
    'hidden_dim': [64, 128, 256],
    'num_attention_heads': [4, 8, 16],
    'num_layers': [3, 5, 10],
    'lr': [0.0001, 0.001, 0.01],
    'weight_decay': [0.0, 0.001, 0.01]
}
```

## Evaluation

The model is evaluated using multiple metrics:
- Accuracy
- Precision
- Recall
- ROC AUC
- Confusion Matrix

Results are saved in `evaluation_metrics.json` after training.
