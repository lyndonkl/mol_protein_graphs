# Standard library imports
import os
import random
import json

# Third-party imports
# Standard library imports
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import traceback
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Custom imports
from datasets import CombinedDataset, MoleculeDataset
from protein_processor import ProteinProcessor
from model import CrossAttentionLayer, StackedCrossGraphAttentionModel
from utils import setup_logger, collect_protein_node_and_edge_types, collate_fn

# Constants
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set environment variables at the top-level scope
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, rank, world_size, graph_metadata):
        self.logger = setup_logger()
        self.logger.info(f"[Rank {rank}] Initializing Trainer")

        self.rank = rank
        self.world_size = world_size
        self.model = model

        # Create a DistributedSampler for the training data
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            num_workers=30,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=collate_fn
        )

        # Ensure model is on the correct device before performing the dummy forward pass
        self.model = self.model.to(self.rank)
        self.logger.info(f"[Rank {rank}] Model moved to device {rank}")

        try:
            self.model = DistributedDataParallel(
                self.model, device_ids=[rank], output_device=rank, find_unused_parameters=True
            )
            self.logger.info(f"[Rank {rank}] Model wrapped with DistributedDataParallel")
        except Exception as e:
            self.logger.error(f"[Rank {rank}] Exception during model wrapping: {e}")
            traceback.print_exc()
            raise e

        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.logger.info(f"[Rank {rank}] Optimizer initialized")

        if rank == 0:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=64,
                num_workers=30,
                shuffle=False,
                collate_fn=collate_fn
            )
            
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=64,
                num_workers=30,
                shuffle=False,
                collate_fn=collate_fn
            )
        
        self.logger.info(f"[Rank {self.rank}] Train loader initialized with {len(self.train_loader)} batches")

    def train_epoch(self, epoch):
        try:
            # Set the epoch for the DistributedSampler
            self.logger.info(f"[Rank {self.rank}] Starting training epoch {epoch}")
            self.train_loader.sampler.set_epoch(epoch)

            self.model.train()
            total_loss = torch.zeros(1).to(self.rank)
            total_samples = torch.zeros(1).to(self.rank)
            for mol_data, prot_data, batch_size in tqdm(self.train_loader, desc="Training", disable=(self.rank != 0)):
                self.optimizer.zero_grad()
                mol_data = mol_data.to(self.rank)
                prot_data = prot_data.to(self.rank)
                out = self.model(mol_data, prot_data)
                y = mol_data['smolecule'].y.to(self.rank)

                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch_size
                total_samples += batch_size

                if batch_size % 100 == 0:
                    self.logger.info(f"[Rank {self.rank}] Processing batch {batch_idx}")

            self.logger.info(f"[Rank {self.rank}] Finished training epoch {epoch}")

            # Perform all_reduce on total_loss and total_samples
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
            self.logger.info(f"[Rank {self.rank}] Completed dist.all_reduce in epoch {epoch}")

            average_loss = total_loss.item() / total_samples.item()
            return average_loss
        except Exception as e:
            self.logger.error(f"[Rank {self.rank}] Error in train_epoch: {e}")
            traceback.print_exc()
            raise e

    def validate(self):
        try:
            if self.rank != 0:
                return None
            
            self.logger.info(f"[Rank {self.rank}] Starting validation")
            self.model.eval()
            total_loss = 0
            total_samples = 0
            with torch.no_grad():
                for mol_data, prot_data, batch_size in tqdm(self.val_loader, desc="Validating"):
                    mol_data = mol_data.to(self.rank)
                    prot_data = prot_data.to(self.rank)
                    out = self.model(mol_data, prot_data)
                    y = mol_data['smolecule'].y.to(self.rank)

                    loss = self.criterion(out, y)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size

            self.logger.info(f"[Rank {self.rank}] Finished validation")

            return total_loss / total_samples
        except Exception as e:
            self.logger.error(f"[Rank {self.rank}] Error in validate: {e}")
            traceback.print_exc()
            raise e

    def test(self):
        try:
            if self.rank != 0:
                return None, None
            
            self.model.eval()
            predictions = []
            true_labels = []
            with torch.no_grad():
                for mol_data, prot_data, batch_size in tqdm(self.test_loader, desc="Testing"):
                    mol_data = mol_data.to(self.rank)
                    prot_data = prot_data.to(self.rank)
                    out = self.model(mol_data, prot_data)
                    y = mol_data['smolecule'].y.to(self.rank)


                    predictions.extend(out.cpu().numpy())
                    true_labels.extend(y.cpu().numpy())

            return predictions, true_labels
        except Exception as e:
            self.logger.error(f"[Rank {self.rank}] Error in test: {e}")
            traceback.print_exc()
            raise e

def run(rank: int, world_size: int, train_dataset, val_dataset, test_dataset, graph_metadata):
    # Set random seeds for reproducibility
    # Set the CUDA device
    torch.cuda.set_device(rank)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    logger = setup_logger()
    logger.info(f"[Rank {rank}] Starting run function")
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    logger.info(f"[Rank {rank}] Process group initialized")

    # Initialize model, criterion, and optimizer
    model = StackedCrossGraphAttentionModel(graph_metadata, hidden_dim=128, num_attention_heads=8, num_layers=4)
    logger.info(f"[Rank {rank}] Model initialized")

    # Train the model
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset, rank, world_size, graph_metadata)

    num_epochs = 3  # Set your number of epochs
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(epoch)

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

        # All processes wait here until validation is complete
        torch.distributed.barrier()


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
    logger.info(f"[Rank {rank}] Destroyed process group and exiting")

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
