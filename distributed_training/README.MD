# Distributed Training for Cross-Graph Attention Model

This project implements a distributed training pipeline for a Cross-Graph Attention Model using PyTorch's DistributedDataParallel. The model is designed to predict molecular binding interactions between proteins and small molecules.

## Project Structure

- `setup_and_run.sh`: Setup script for installing dependencies and running the training script.
- `training.py`: Main script for distributed training of the model.
- `model.py`: Contains the implementation of the Cross-Graph Attention Model.
- `datasets.py`: Defines custom datasets for molecules and proteins.
- `protein_processor.py`: Processes protein data for input to the model.
- `utils.py`: Utility functions for data processing and logging.

## Setup and Running

1. Make the setup script executable:
   ```
   chmod +x setup_and_run.sh
   ```

2. Run the setup script:
   ```
   ./setup_and_run.sh
   ```

   This script will:
   - Update pip
   - Install required Python packages
   - Detect CUDA version and install appropriate PyTorch version
   - Run the training script

3. To run on a server, you may need to use a job scheduling system. For example, with SLURM:
   ```
   sbatch submit_job.sh
   ```

   Where `submit_job.sh` might look like:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=distributed_training
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=4
   #SBATCH --gres=gpu:4

   ./setup_and_run.sh
   ```

## Code Overview

### training.py

This script orchestrates the distributed training process:
- Sets up the distributed environment
- Initializes the model, datasets, and data loaders
- Implements the training, validation, and testing loops
- Handles model saving and performance evaluation

### model.py

Defines the Cross-Graph Attention Model:
- `CrossAttentionLayer`: Implements cross-attention mechanism
- `CrossGraphAttentionModel`: Main model architecture combining graph convolutions and cross-attention

### datasets.py

Contains custom dataset implementations:
- `MoleculeDataset`: Processes molecular data
- `CombinedDataset`: Combines molecule and protein data

### protein_processor.py

Handles protein data processing:
- Converts protein sequences to graph representations

### utils.py

Provides utility functions:
- `custom_transform`: Prepares batch data for model input
- `collate_fn`: Processes and combines batch items
- `setup_logger`: Configures logging for the training process

## Model Architecture

The Cross-Graph Attention Model consists of:
1. Separate graph convolution layers for molecules and proteins
2. Cross-attention layers to capture interactions between molecules and proteins
3. Global pooling and fully connected layers for final prediction

## Training Process

The distributed training process involves:
1. Data parallelism using DistributedDataParallel
2. Epoch-wise training with validation
3. Model saving based on best validation performance
4. Final testing and performance evaluation

## Performance Metrics

The model's performance is evaluated using:
- Accuracy
- ROC-AUC
- Precision
- Recall
- F1-Score

For any issues or questions, please refer to the documentation or contact the project maintainers.
