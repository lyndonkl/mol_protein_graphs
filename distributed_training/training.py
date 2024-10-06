# Standard library imports
import os
import random
import json
from typing import List, Tuple, Dict

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Dataset, HeteroData, Batch
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import random_split
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import three_to_index, index_to_one
from sklearn.model_selection import train_test_split
import logging
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Custom imports
from datasets import CombinedDataset, MoleculeDataset
from protein_processor import ProteinProcessor
from model import CrossAttentionLayer, CrossGraphAttentionModel
from utils import custom_transform, collate_fn, setup_logger

# Constants
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, criterion, optimizer, rank, world_size):
        self.model = DDP(model.to(rank), device_ids=[rank])
        self.rank = rank
        self.world_size = world_size
        self.criterion = criterion
        self.optimizer = optimizer

        # Combine molecule and protein node types
        all_node_types = molecule_node_types + protein_node_types

        # Create input_nodes dictionary for training
        train_size = len(train_dataset)
        indices = torch.arange(train_size)
        split = train_size // world_size
        start_idx = rank * split
        end_idx = start_idx + split if rank != (world_size - 1) else train_size
        train_indices = indices[start_idx:end_idx]
        train_input_nodes = {node_type: train_indices for node_type in all_node_types}

        # Create num_samples dictionary
        num_samples = {node_type: [10, 5] for node_type in all_node_types}

        self.train_loader = HGTLoader(
            train_dataset,
            num_samples=num_samples,
            input_nodes=train_input_nodes,
            batch_size=64,
            num_workers=4,
            shuffle=True,
            transform=custom_transform
        )

        if rank == 0:
            val_input_nodes = {node_type: None for node_type in all_node_types}
            self.val_loader = HGTLoader(
                val_dataset,
                num_samples=num_samples,
                input_nodes=val_input_nodes,
                batch_size=64,
                num_workers=4,
                shuffle=False,
                transform=custom_transform
            )
            self.test_loader = HGTLoader(
                test_dataset,
                num_samples=num_samples,
                input_nodes=val_input_nodes,
                batch_size=64,
                num_workers=4,
                shuffle=False,
                transform=custom_transform
            )

    def train_epoch(self):
        self.model.train()
        total_loss = torch.zeros(2).to(self.rank)
        for batch in tqdm(self.train_loader, desc="Training", disable=(self.rank != 0)):
            self.optimizer.zero_grad()
            mol_data = batch['mol_batch'].to(self.rank)
            prot_data = batch['prot_batch'].to(self.rank)
            batch_size = batch['batch_size']
            out = self.model(mol_data, prot_data)
            
            out = out[:batch_size]
            y = mol_data['smolecule'].y[:batch_size].to(self.rank)

            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            total_loss[0] += float(loss) * batch_size
            total_loss[1] += batch_size

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        return float(total_loss[0] / total_loss[1])

    def validate(self):
        if self.rank != 0:
            return None
        
        self.model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                mol_data = batch['mol_batch'].to(self.rank)
                prot_data = batch['prot_batch'].to(self.rank)
                batch_size = batch['batch_size']
                out = self.model(mol_data, prot_data)
                out = out[:batch_size]
                y = mol_data['smolecule'].y[:batch_size].to(self.rank)
                loss = self.criterion(out, y)
                total_loss += float(loss) * batch_size
                total_samples += batch_size
        return total_loss / total_samples

    def test(self):
        if self.rank != 0:
            return None, None
        
        self.model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                mol_data = batch['mol_batch'].to(self.rank)
                prot_data = batch['prot_batch'].to(self.rank)
                batch_size = batch['batch_size']
                out = self.model(mol_data, prot_data)
                out = out[:batch_size]
                predictions.extend(out.cpu().numpy())
                true_labels.extend(mol_data['smolecule'].y[:batch_size].cpu().numpy())
        return predictions, true_labels

def run(rank: int, world_size: int, train_dataset, val_dataset, test_dataset, graph_metadata):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    logger = setup_logger() if rank == 0 else None

    # Initialize model, criterion, and optimizer
    model = CrossGraphAttentionModel(hidden_dim=64, num_attention_heads=4).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset, criterion, optimizer, rank, world_size)

    num_epochs = 5  # Set your number of epochs
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch()

        # Add barrier to synchronize all processes
        torch.distributed.barrier()

        if rank == 0:
            val_loss = trainer.validate()
            logger.info(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save the model if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the model
                torch.save(model.module.state_dict(), 'best_cross_graph_attention_model.pth')
                logger.info(f'New best model saved at epoch {epoch}')

    if rank == 0:
        test_predictions, test_true = trainer.test()
        
        # Apply a threshold to obtain binary predictions
        threshold = 0.5
        test_pred_binary = [1 if p >= threshold else 0 for p in test_predictions]

        # Evaluate performance
        accuracy = accuracy_score(test_true, test_pred_binary)
        roc_auc = roc_auc_score(test_true, test_predictions)
        precision = precision_score(test_true, test_pred_binary)
        recall = recall_score(test_true, test_pred_binary)
        f1 = f1_score(test_true, test_pred_binary)

        logger.info("Test Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")

    dist.destroy_process_group()

def main():
    # Load and preprocess data
    df = pd.read_parquet('cleaned_train.parquet')
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['binds'], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['binds'], random_state=RANDOM_SEED)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # Process proteins
    protein_pdb_files = {
        'BRD4': './BRD4.pdb',
        'HSA': './ALB.pdb',
        'sEH': './EPH.pdb'
    }
    protein_graphs = {protein_name: ProteinProcessor.process_protein(pdb_file) 
                      for protein_name, pdb_file in protein_pdb_files.items() if os.path.exists(pdb_file)}

    # Load unique atom and edge types
    with open('unique_atom_and_edge_types.json', 'r') as f:
        unique_types = json.load(f)

    molecule_node_types, molecule_edge_types, protein_node_types, protein_edge_types
    molecule_node_types = unique_types['molecule_node_types']
    molecule_edge_types = [tuple(edge) for edge in unique_types['molecule_edge_types']]
    protein_node_types, protein_edge_types = collect_protein_node_and_edge_types(protein_graphs)
    graph_metadata = {
        'molecule_node_types': molecule_node_types,
        'molecule_edge_types': molecule_edge_types,
        'protein_node_types': protein_node_types,
        'protein_edge_types': protein_edge_types
    }

    # Create datasets and data loaders
    train_dataset = CombinedDataset(train_df, protein_graphs)
    val_dataset = CombinedDataset(val_df, protein_graphs)
    test_dataset = CombinedDataset(test_df, protein_graphs)

    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, train_dataset, val_dataset, test_dataset, graph_metadata), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
