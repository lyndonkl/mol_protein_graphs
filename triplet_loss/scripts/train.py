# Standard library imports
import random
import json
import torch.multiprocessing as mp

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split

# Custom imports
from triplet_loss.src.data.datasets import CombinedDataset
from triplet_loss.src.model.model import StackedMoleculeGraphTripletModel
from triplet_loss.src.training.train import TripletTrainer
from triplet_loss.src.training.evaluator import TripletEvaluator
from triplet_loss.src.utils.helpers import setup_logger

# Constants
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(rank: int, world_size: int, train_dataset, val_dataset, test_dataset, graph_metadata):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    logger = setup_logger()
    logger.info(f"[Rank {rank}] Starting run function")
    
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    logger.info(f"[Rank {rank}] Process group initialized with backend {backend}")

    model = StackedMoleculeGraphTripletModel(3, graph_metadata, hidden_dim=256, num_attention_heads=16, num_layers=3)
    model.train()  # Make sure model is in training mode
    
    logger.info(f"[Rank {rank}] Model initialized")

    optimizer_params = {
        'lr': 0.0001,
        'weight_decay': 0.001
    }

    trainer = TripletTrainer(model, train_dataset, val_dataset, rank, world_size, optimizer_params)

    num_epochs = 5
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(epoch)

        dist.barrier()

        val_loss = trainer.validate()
        logger.info(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'triplet_loss/data/best_triplet_model.pth')
                logger.info(f'New best model saved at epoch {epoch}')

        trainer.scheduler.step()

        dist.barrier()

    if rank == 0:
        logger.info("Starting evaluation...")
    
    # Need to unwrap the model from DDP before evaluation
    model = trainer.model.module
    model.load_state_dict(torch.load('triplet_loss/data/best_triplet_model.pth'))
    
    evaluator = TripletEvaluator(
        model=model,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        rank=rank,
        world_size=world_size,
        batch_size=32,
        device=DEVICE
    )
    
    evaluator.evaluate()
    
    if rank == 0:
        logger.info("Evaluation complete!")
        
    dist.destroy_process_group()
    logger.info(f"[Rank {rank}] Destroyed process group and exiting")

def main():
    df = pd.read_parquet('triplet_loss/data/cleaned_train_unique.parquet')
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    with open('triplet_loss/data/unique_atom_and_edge_types.json', 'r') as f:
        unique_types = json.load(f)

    molecule_node_types = unique_types['molecule_node_types']
    molecule_edge_types = [tuple(edge) for edge in unique_types['molecule_edge_types']]
    graph_metadata = {
        'molecule_node_types': molecule_node_types,
        'molecule_edge_types': molecule_edge_types
    }

    train_dataset = CombinedDataset(train_df)
    val_dataset = CombinedDataset(val_df)
    test_dataset = CombinedDataset(test_df)

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()

    mp.spawn(run, args=(world_size, train_dataset, val_dataset, test_dataset, graph_metadata), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()