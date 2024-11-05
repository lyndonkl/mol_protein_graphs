# Standard library imports
import random
import json
import torch.multiprocessing as mp

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

# Custom imports
from triplet_loss.src.data.datasets import CombinedDataset
from triplet_loss.src.model.model import StackedMoleculeGraphTripletModel
from triplet_loss.src.training.evaluator import TripletEvaluator
from triplet_loss.src.utils.helpers import setup_logger

# Constants
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(rank: int, world_size: int, val_dataset, test_dataset, graph_metadata):
    """
    Run the prediction process on a single GPU.
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        val_dataset: Dataset for generating reference embeddings
        test_dataset: Dataset to generate predictions for
        graph_metadata: Metadata about the graph structure
    """
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    logger = setup_logger()
    logger.info(f"[Rank {rank}] Starting prediction process")
    
    # Initialize process group
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    logger.info(f"[Rank {rank}] Process group initialized with backend {backend}")

    # Initialize model
    model = StackedMoleculeGraphTripletModel(
        3, 
        graph_metadata, 
        hidden_dim=256, 
        num_attention_heads=16, 
        num_layers=3
    )
    model.eval()  # Set model to evaluation mode
    
    # Load trained model weights
    model_path = 'triplet_loss/data/best_triplet_model.pth'
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))
    
    logger.info(f"[Rank {rank}] Model loaded from {model_path}")

    # Initialize evaluator
    evaluator = TripletEvaluator(
        model=model,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        rank=rank,
        world_size=world_size,
        batch_size=4,
        device=DEVICE
    )
    
    # Generate predictions
    evaluator.predict()
    
    if rank == 0:
        logger.info("Predictions complete and saved to results/binding_predictions.csv")
        
    dist.destroy_process_group()
    logger.info(f"[Rank {rank}] Destroyed process group and exiting")

def main():
    """Main function to set up and run the prediction process."""
    # Load datasets
    val_df = pd.read_parquet('triplet_loss/data/cleaned_train_unique.parquet')
    test_df = pd.read_parquet('triplet_loss/data/cleaned_test.parquet')

    print(f"Reference set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # Load graph metadata
    with open('triplet_loss/data/unique_atom_and_edge_types.json', 'r') as f:
        unique_types = json.load(f)

    molecule_node_types = unique_types['molecule_node_types']
    molecule_edge_types = [tuple(edge) for edge in unique_types['molecule_edge_types']]
    graph_metadata = {
        'molecule_node_types': molecule_node_types,
        'molecule_edge_types': molecule_edge_types
    }

    # Create datasets
    val_dataset = CombinedDataset(val_df, predicting=False)
    test_dataset = CombinedDataset(test_df, predicting=True)

    # Determine number of processes based on available GPUs or CPU cores
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()

    # Spawn processes
    mp.spawn(
        run,
        args=(world_size, val_dataset, test_dataset, graph_metadata),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main() 