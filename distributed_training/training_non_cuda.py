# Standard library imports
import os
import random
import json

# Third-party imports
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
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
DEVICE = torch.device('cpu')

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set environment variables at the top-level scope
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.multiprocessing.set_sharing_strategy('file_system')

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
            batch_size=8,
            num_workers=2,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=collate_fn
        )

        # Ensure model is on the correct device before performing the dummy forward pass
        self.model = self.model.to(DEVICE)
        self.logger.info(f"[Rank {rank}] Model moved to device {DEVICE}")

        try:
            self.model = DistributedDataParallel(self.model, device_ids=None, find_unused_parameters=True)
            self.logger.info(f"[Rank {rank}] Model wrapped with DistributedDataParallel")
        except Exception as e:
            self.logger.error(f"[Rank {rank}] Exception during model wrapping: {e}")
            traceback.print_exc()
            raise e

        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.logger.info(f"[Rank {rank}] Optimizer initialized")


        self.val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            num_workers=2,
            shuffle=False,
            sampler=self.val_sampler,
            collate_fn=collate_fn
        )

        self.test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            num_workers=2,
            shuffle=False,
            sampler=self.test_sampler,
            collate_fn=collate_fn
        )
        
        self.logger.info(f"[Rank {self.rank}] Train loader initialized with {len(self.train_loader)} batches")

        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)  # Decay LR by 10% every epoch

    def train_epoch(self, epoch):
        try:
            # Set the epoch for the DistributedSampler
            self.logger.info(f"[Rank {self.rank}] Starting training epoch {epoch}")
            self.train_loader.sampler.set_epoch(epoch)

            self.model.train()
            total_loss = 0.0
            total_samples = 0
            accumulated_loss = 0.0
            accumulation_steps = 16  # Simulate batch size of 64 (8 * 8)
            self.optimizer.zero_grad()  # Zero gradients at the beginning of epoch

            for i, (mol_data, prot_data, batch_size) in enumerate(tqdm(self.train_loader, desc="Training", disable=(self.rank != 0))):
                mol_data = mol_data.to(DEVICE)
                prot_data = prot_data.to(DEVICE)
                out = self.model(mol_data, prot_data)
                y = mol_data['smolecule'].y.to(DEVICE)

                loss = self.criterion(out, y)
                loss = loss / accumulation_steps  # Normalize the loss
                loss.backward()
                accumulated_loss += loss.item() * batch_size

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_loss += accumulated_loss
                    accumulated_loss = 0.0

                total_samples += batch_size

            self.logger.info(f"[Rank {self.rank}] Finished training epoch {epoch}")

            if DEVICE.type == 'cpu':
                # Gather losses from all processes only if using CPU (multi-process)
                all_losses = [torch.zeros(1).to(DEVICE) for _ in range(self.world_size)]
                all_samples = [torch.zeros(1, dtype=torch.long).to(DEVICE) for _ in range(self.world_size)]
                
                dist.all_gather(all_losses, torch.tensor([total_loss]).to(DEVICE))
                dist.all_gather(all_samples, torch.tensor([total_samples]).to(DEVICE))

                total_loss = sum(loss.item() for loss in all_losses)
                total_samples = sum(samples.item() for samples in all_samples)
            else:
                # For MPS, we're already using a single process, so no need to gather
                pass

            average_loss = total_loss / total_samples
            return average_loss
        except Exception as e:
            self.logger.error(f"[Rank {self.rank}] Error in train_epoch: {e}")
            traceback.print_exc()
            raise e

    def validate(self):
        try:
            self.logger.info(f"[Rank {self.rank}] Starting validation")
            self.model.eval()
            total_loss = 0
            total_samples = 0
            with torch.no_grad():
                for mol_data, prot_data, batch_size in tqdm(self.val_loader, desc="Validating", disable=(self.rank != 0)):
                    mol_data = mol_data.to(DEVICE)
                    prot_data = prot_data.to(DEVICE)
                    out = self.model(mol_data, prot_data)
                    y = mol_data['smolecule'].y.to(DEVICE)
                    
                    loss = self.criterion(out, y)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size

            self.logger.info(f"[Rank {self.rank}] Finished validation")

            if DEVICE.type == 'cpu':
                # Gather losses from all processes only if using CPU (multi-process)
                all_losses = [torch.zeros(1).to(DEVICE) for _ in range(self.world_size)]
                all_samples = [torch.zeros(1, dtype=torch.long).to(DEVICE) for _ in range(self.world_size)]
                
                dist.all_gather(all_losses, torch.tensor([total_loss]).to(DEVICE))
                dist.all_gather(all_samples, torch.tensor([total_samples]).to(DEVICE))

                total_loss = sum(loss.item() for loss in all_losses)
                total_samples = sum(samples.item() for samples in all_samples)
            else:
                # For MPS, we're already using a single process, so no need to gather
                pass

            average_loss = total_loss / total_samples
            return average_loss
        except Exception as e:
            self.logger.error(f"[Rank {self.rank}] Error in validate: {e}")
            traceback.print_exc()
            raise e

    def test(self):
        try:
            self.logger.info(f"[Rank {self.rank}] Starting testing")
            self.model.eval()
            predictions = []
            true_labels = []
            with torch.no_grad():
                for mol_data, prot_data, batch_size in tqdm(self.test_loader, desc="Testing"):
                    mol_data = mol_data.to(DEVICE)
                    prot_data = prot_data.to(DEVICE)
                    out = self.model(mol_data, prot_data)
                    y = mol_data['smolecule'].y.to(DEVICE)

                    predictions.extend(out.cpu().numpy())
                    true_labels.extend(y.cpu().numpy())

            self.logger.info(f"[Rank {self.rank}] Finished testing")

            # Gather predictions and true labels from all processes
            all_predictions = [torch.zeros_like(torch.tensor(predictions), device=DEVICE) for _ in range(self.world_size)]
            all_true_labels = [torch.zeros_like(torch.tensor(true_labels), device=DEVICE) for _ in range(self.world_size)]

            dist.all_gather(all_predictions, torch.tensor(predictions))
            dist.all_gather(all_true_labels, torch.tensor(true_labels))

            all_predictions = torch.cat(all_predictions, dim=0)
            all_true_labels = torch.cat(all_true_labels, dim=0)

            return all_predictions, all_true_labels
        except Exception as e:
            self.logger.error(f"[Rank {self.rank}] Error in test: {e}")
            traceback.print_exc()
            raise e

def run(rank: int, world_size: int, train_dataset, val_dataset, test_dataset, graph_metadata):
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    logger = setup_logger()
    logger.info(f"[Rank {rank}] Starting run function")
    
    # Use 'gloo' backend for CPU and MPS
    backend = 'gloo'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    logger.info(f"[Rank {rank}] Process group initialized with backend {backend}")

    # Initialize model, criterion, and optimizer
    model = StackedCrossGraphAttentionModel(graph_metadata, hidden_dim=128, num_attention_heads=8, num_layers=4)
    logger.info(f"[Rank {rank}] Model initialized")

    # Train the model
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset, rank, world_size, graph_metadata)

    num_epochs = 5  # Set your number of epochs
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(epoch)

        # Add barrier to synchronize all processes
        torch.distributed.barrier()

        val_loss = trainer.validate()
        logger.info(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save the model if it's the best so far
        if rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the model
                torch.save(model.state_dict(), 'best_cross_graph_attention_model.pth')
                logger.info(f'New best model saved at epoch {epoch}')

        trainer.scheduler.step()

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
    df = pd.read_parquet('../cleaned_train.parquet')
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['binds'], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['binds'], random_state=RANDOM_SEED)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # Process proteins
    protein_pdb_files = {
        'BRD4': '../BRD4.pdb',
        'HSA': '../ALB.pdb',
        'sEH': '../EPH.pdb'
    }
    protein_graphs = {protein_name: ProteinProcessor.process_protein(pdb_file) 
                      for protein_name, pdb_file in protein_pdb_files.items() if os.path.exists(pdb_file)}

    # Load unique atom and edge types
    with open('../unique_atom_and_edge_types.json', 'r') as f:
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

    # For CPU, use the number of available cores. For MPS, use 1.
    world_size = mp.cpu_count() if DEVICE.type == 'cpu' else 1

    mp.spawn(run, args=(world_size, train_dataset, val_dataset, test_dataset, graph_metadata), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
