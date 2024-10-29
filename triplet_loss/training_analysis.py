import time
import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import ParameterGrid, train_test_split
from tqdm import tqdm
import warnings

# Custom imports
from datasets import CombinedDataset
from model import StackedMoleculeGraphTripletModel
from utils import setup_logger, collate_fn

# Constants
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ignore all warnings
warnings.filterwarnings("ignore")

def determine_dataset_size(dataset, batch_size, model, optimizer, criterion, max_time_minutes=10):
    logger = setup_logger()
    logger.info("Starting dataset size determination")
    
    dataset_sizes = []
    training_times = []
    
    # Initial size and increment
    dataset_size = 1000
    increment = 500
    
    while True:
        # Create a subset of the dataset
        subset = Subset(dataset, range(min(dataset_size, len(dataset))))
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        # Measure training time for one epoch
        model.train()
        start_time = time.time()
        for anchor, positive, negative, protein_type_id, _ in tqdm(dataloader, desc="Training", leave=False):
            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)
            protein_type_id = protein_type_id.to(DEVICE)

            anchor_out = model(anchor, protein_type_id)
            positive_out = model(positive, protein_type_id)
            negative_out = model(negative, protein_type_id)

            loss = criterion(anchor_out, positive_out, negative_out)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        training_time = (end_time - start_time) / 60  # Convert to minutes
        
        dataset_sizes.append(dataset_size)
        training_times.append(training_time)
        
        logger.info(f"Dataset size: {dataset_size}, Training time (minutes): {training_time:.2f}")
        
        # Log the training time
        logger.info(f"Training time for dataset size {dataset_size}: {training_time:.2f} minutes")

        # Check if training time exceeds the maximum allowed time
        if training_time > max_time_minutes:
            logger.info(f"Training time exceeded {max_time_minutes} minutes. Stopping dataset size determination.")
            break
        
        # Increment the dataset size
        dataset_size += increment
    
    # Plot dataset size vs training time
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, training_times, marker='o')
    plt.xlabel('Dataset Size')
    plt.ylabel('Training Time (minutes)')
    plt.title('Dataset Size vs Training Time')
    plt.savefig('dataset_size_vs_training_time.png')
    plt.close()
    
    # Return the maximum dataset size that completes in under max_time_minutes
    for size, time_taken in zip(dataset_sizes, training_times):
        if time_taken <= max_time_minutes:
            optimal_size = size
        else:
            break
    
    logger.info(f"Optimal dataset size determined: {optimal_size}")
    return optimal_size

def run_grid_search(dataset, model_class, param_grid, batch_size, max_dataset_size, graph_metadata):
    logger = setup_logger()
    logger.info("Starting grid search")
    
    # Create a subset of the dataset for grid search
    subset = Subset(dataset, range(min(max_dataset_size, len(dataset))))
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    best_loss = float('inf')
    best_params = None
    
    # Separate model parameters from optimizer parameters
    for params in ParameterGrid(param_grid):
        logger.info(f"Trying parameters: {params}")
        
        # Extract optimizer parameters
        optimizer_params = {
            'lr': params.pop('lr'),
            'weight_decay': params.pop('weight_decay')
        }
        
        # Create model with only model parameters
        model = model_class(3, graph_metadata, **params).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        criterion = torch.nn.TripletMarginLoss(margin=1.0)
        
        model.train()
        total_loss = 0
        num_samples = 0
        
        for anchor, positive, negative, protein_type_id, actual_batch_size in tqdm(dataloader, desc="Training"):
            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)
            protein_type_id = protein_type_id.to(DEVICE)

            anchor_out = model(anchor, protein_type_id)
            positive_out = model(positive, protein_type_id)
            negative_out = model(negative, protein_type_id)

            loss = criterion(anchor_out, positive_out, negative_out)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * actual_batch_size
            num_samples += actual_batch_size
        
        avg_loss = total_loss / num_samples
        logger.info(f"Average loss for parameters {params}: {avg_loss}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = params
    
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best loss: {best_loss}")
    
    return best_params, best_loss

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    logger = setup_logger()
    
    # Load data
    df = pd.read_parquet('cleaned_train_unique.parquet')
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)

    logger.info(f"Training set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}")

    # Load graph metadata
    with open('unique_atom_and_edge_types.json', 'r') as f:
        unique_types = json.load(f)

    molecule_node_types = unique_types['molecule_node_types']
    molecule_edge_types = [tuple(edge) for edge in unique_types['molecule_edge_types']]
    graph_metadata = {
        'molecule_node_types': molecule_node_types,
        'molecule_edge_types': molecule_edge_types
    }

    # Create dataset
    train_dataset = CombinedDataset(train_df)

    # Initialize model for dataset size determination
    # initial_model = StackedMoleculeGraphTripletModel(3, graph_metadata, hidden_dim=128, num_attention_heads=8, num_layers=4).to(DEVICE)
    # initial_optimizer = torch.optim.Adam(initial_model.parameters(), lr=1e-3)
    # initial_criterion = torch.nn.TripletMarginLoss(margin=1.0)

    # Determine optimal dataset size
    # max_dataset_size = determine_dataset_size(train_dataset, batch_size=8, model=initial_model, optimizer=initial_optimizer, criterion=initial_criterion)
    max_dataset_size = 4000 # From Previous Analysis

    # Define parameter grid for grid search
    param_grid = {
        'hidden_dim': [64, 128, 256],
        'num_attention_heads': [4, 8, 16],
        'num_layers': [3, 5, 10],
        'lr': [0.0001, 0.001, 0.01],
        'weight_decay': [0.0, 0.001, 0.01]
    }

    # Run grid search
    best_params, best_loss = run_grid_search(train_dataset, StackedMoleculeGraphTripletModel, param_grid, batch_size=8, max_dataset_size=max_dataset_size, graph_metadata=graph_metadata)

    logger.info("Training analysis completed")
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best loss: {best_loss}")

if __name__ == "__main__":
    main()
