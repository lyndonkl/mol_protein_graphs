# Leash-Belka: Distributed Training for Cross-Graph Attention Model

This project implements a distributed training pipeline for a Cross-Graph Attention Model using PyTorch's DistributedDataParallel. The model is designed to predict molecular binding interactions between proteins and small molecules.

## Project Structure

The project is organized into the following main directories:

- `distributed_training/`: Contains the core implementation of the distributed training pipeline.
- `data/`: (Assumed) Directory for storing input data files.
- `models/`: (Assumed) Directory for saving trained models.

## Key Components

### Distributed Training

Located in the `distributed_training/` directory:

- `setup_and_run.sh`: Setup script for installing dependencies and running the training script.
- `training.py`: Main script for distributed training of the model.
- `model.py`: Contains the implementation of the Cross-Graph Attention Model.
- `datasets.py`: Defines custom datasets for molecules and proteins.
- `protein_processor.py`: Processes protein data for input to the model.
- `utils.py`: Utility functions for data processing and logging.

## Setup and Running

1. Navigate to the `distributed_training/` directory.

2. Make the setup script executable:
   ```
   chmod +x setup_and_run.sh
   ```

3. Run the setup script:
   ```
   ./setup_and_run.sh
   ```

   This script will:
   - Update pip
   - Install required Python packages
   - Detect CUDA version and install appropriate PyTorch version
   - Run the training script

4. For running on a server with a job scheduling system like SLURM, you may need to create and use a job submission script.

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

For more detailed information about each component, please refer to the README files in the respective directories.

For any issues or questions, please refer to the documentation or contact the project maintainers.

