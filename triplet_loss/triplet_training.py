# Standard library imports
import os
import random
import json

# Third-party imports
import warnings
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import pickle

# Custom imports
from datasets import CombinedDataset
from model import StackedMoleculeGraphTripletModel
from utils import setup_logger, collate_fn

# Constants
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set environment variables
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.multiprocessing.set_sharing_strategy('file_system')

class TripletTrainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, rank, world_size, graph_metadata):
        self.logger = setup_logger()
        self.logger.info(f"[Rank {rank}] Initializing TripletTrainer")

        self.rank = rank
        self.world_size = world_size
        self.model = model

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=self.rank, shuffle=True)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            num_workers=2,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=collate_fn
        )

        self.model = self.model.to(DEVICE)
        self.logger.info(f"[Rank {rank}] Model moved to device {self.rank}")

        model_path = 'best_triplet_model.pth'
        if os.path.exists(model_path):
            self.logger.info(f"[Rank {rank}] Loading existing model weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        try:
            self.model = DistributedDataParallel(self.model, device_ids=None, find_unused_parameters=True)
            self.logger.info(f"[Rank {rank}] Model wrapped with DistributedDataParallel")
        except Exception as e:
            self.logger.error(f"[Rank {rank}] Exception during model wrapping: {e}")
            traceback.print_exc()
            raise e

        self.criterion = torch.nn.TripletMarginLoss(margin=1.0)
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

        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)

    def train_epoch(self, epoch):
        try:
            self.logger.info(f"[Rank {self.rank}] Starting training epoch {epoch}")
            self.train_loader.sampler.set_epoch(epoch)

            self.model.train()
            total_loss = 0.0
            total_samples = 0
            accumulated_loss = 0.0
            accumulation_steps = 16
            self.optimizer.zero_grad()

            for i, (anchor, positive, negative, protein_type_id, batch_size) in enumerate(tqdm(self.train_loader, desc="Training", disable=(self.rank != 0))):
                anchor = anchor.to(DEVICE)
                positive = positive.to(DEVICE)
                negative = negative.to(DEVICE)
                protein_type_id = protein_type_id.to(DEVICE)

                anchor_out = self.model(anchor, protein_type_id)
                positive_out = self.model(positive, protein_type_id)
                negative_out = self.model(negative, protein_type_id)

                loss = self.criterion(anchor_out, positive_out, negative_out)
                loss = loss / accumulation_steps
                loss.backward()
                accumulated_loss += loss.item() * batch_size

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_loss += accumulated_loss
                    accumulated_loss = 0.0

                total_samples += batch_size

            self.logger.info(f"[Rank {self.rank}] Finished training epoch {epoch}")

            all_losses = [torch.zeros(1).to(DEVICE) for _ in range(self.world_size)]
            all_samples = [torch.zeros(1, dtype=torch.long).to(DEVICE) for _ in range(self.world_size)]
            
            dist.all_gather(all_losses, torch.tensor([total_loss]).to(DEVICE))
            dist.all_gather(all_samples, torch.tensor([total_samples]).to(DEVICE))

            total_loss = sum(loss.item() for loss in all_losses)
            total_samples = sum(samples.item() for samples in all_samples)

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
                for anchor, positive, negative, protein_type_id, batch_size in tqdm(self.val_loader, desc="Validating", disable=(self.rank != 0)):
                    anchor = anchor.to(DEVICE)
                    positive = positive.to(DEVICE)
                    negative = negative.to(DEVICE)
                    protein_type_id = protein_type_id.to(DEVICE)

                    anchor_out = self.model(anchor, protein_type_id)
                    positive_out = self.model(positive, protein_type_id)
                    negative_out = self.model(negative, protein_type_id)

                    loss = self.criterion(anchor_out, positive_out, negative_out)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size

            self.logger.info(f"[Rank {self.rank}] Finished validation")

            all_losses = [torch.zeros(1).to(DEVICE) for _ in range(self.world_size)]
            all_samples = [torch.zeros(1, dtype=torch.long).to(DEVICE) for _ in range(self.world_size)]
            
            dist.all_gather(all_losses, torch.tensor([total_loss]).to(DEVICE))
            dist.all_gather(all_samples, torch.tensor([total_samples]).to(DEVICE))

            total_loss = sum(loss.item() for loss in all_losses)
            total_samples = sum(samples.item() for samples in all_samples)

            average_loss = total_loss / total_samples
            return average_loss
        except Exception as e:
            self.logger.error(f"[Rank {self.rank}] Error in validate: {e}")
            traceback.print_exc()
            raise e

class TripletEvaluator:
    def __init__(self, model, val_dataset, test_dataset, rank, world_size, batch_size=32, device='cuda'):
        self.model = model
        self.model = self.model.to(device)
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.logger = setup_logger()
        
        # Wrap model in DDP
        try:
            self.model = DistributedDataParallel(self.model, device_ids=None, find_unused_parameters=True)
            self.logger.info(f"[Rank {rank}] Evaluation model wrapped with DistributedDataParallel")
        except Exception as e:
            self.logger.error(f"[Rank {rank}] Exception during model wrapping: {e}")
            traceback.print_exc()
            raise e

        # Create distributed samplers for validation and test datasets
        self.val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        self.test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=2,
            shuffle=False,
            sampler=self.val_sampler,
            collate_fn=collate_fn
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=2,
            shuffle=False,
            sampler=self.test_sampler,
            collate_fn=collate_fn
        )
        
        self.reference_path = 'reference_embeddings.pkl'

    def generate_reference_embeddings(self):
        if self.rank == 0:
            self.logger.info("Generating reference embeddings from validation set...")
            
        self.model.eval()
        binding_embeddings = []
        non_binding_embeddings = []
        
        with torch.no_grad():
            for anchor, positive, negative, protein_type_id, _ in tqdm(self.val_loader, 
                                                                     desc="Generating embeddings",
                                                                     disable=(self.rank != 0)):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                protein_type_id = protein_type_id.to(self.device)
                
                anchor_emb = self.model(anchor, protein_type_id)
                positive_emb = self.model(positive, protein_type_id)
                negative_emb = self.model(negative, protein_type_id)
                
                non_binding_embeddings.append(anchor_emb)
                non_binding_embeddings.append(positive_emb)
                binding_embeddings.append(negative_emb)
        
        binding_embeddings = torch.cat(binding_embeddings)
        non_binding_embeddings = torch.cat(non_binding_embeddings)
        
        gathered_binding = [torch.zeros_like(binding_embeddings) for _ in range(self.world_size)]
        gathered_non_binding = [torch.zeros_like(non_binding_embeddings) for _ in range(self.world_size)]
        
        dist.all_gather(gathered_binding, binding_embeddings)
        dist.all_gather(gathered_non_binding, non_binding_embeddings)
        
        if self.rank == 0:
            all_binding = torch.cat(gathered_binding).cpu().numpy()
            all_non_binding = torch.cat(gathered_non_binding).cpu().numpy()
            
            reference_data = {
                'binding': all_binding,
                'non_binding': all_non_binding
            }
            
            with open(self.reference_path, 'wb') as f:
                pickle.dump(reference_data, f)
            
            self.logger.info("Reference embeddings generated and saved.")
        
        dist.barrier()

    def load_reference_embeddings(self):
        with open(self.reference_path, 'rb') as f:
            return pickle.load(f)
            
    def predict_binding(self, query_embedding, reference_embeddings, k=5):
        if torch.is_tensor(query_embedding):
            query_embedding = query_embedding.cpu().numpy()
        
        binding_distances = np.linalg.norm(reference_embeddings['binding'] - query_embedding, axis=1)
        non_binding_distances = np.linalg.norm(reference_embeddings['non_binding'] - query_embedding, axis=1)
        
        k_nearest_binding = np.partition(binding_distances, k)[:k]
        k_nearest_non_binding = np.partition(non_binding_distances, k)[:k]
        
        avg_binding_dist = np.mean(k_nearest_binding)
        avg_non_binding_dist = np.mean(k_nearest_non_binding)
        
        distances = np.array([avg_non_binding_dist, avg_binding_dist])
        probabilities = F.softmax(torch.tensor(-distances), dim=0).numpy()
        
        return probabilities[1]

    def evaluate(self):
        if self.rank == 0:
            self.logger.info("Starting evaluation...")
        
        if not os.path.exists(self.reference_path):
            self.generate_reference_embeddings()
        reference_embeddings = self.load_reference_embeddings()
        
        self.model.eval()
        true_labels = []
        predicted_probs = []
        
        with torch.no_grad():
            for anchor, positive, negative, protein_type_id, _ in tqdm(self.test_loader, 
                                                                     desc="Evaluating",
                                                                     disable=(self.rank != 0)):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                protein_type_id = protein_type_id.to(self.device)
                
                anchor_emb = self.model(anchor, protein_type_id)
                positive_emb = self.model(positive, protein_type_id)
                negative_emb = self.model(negative, protein_type_id)
                
                for emb, is_binding in [
                    (anchor_emb, False),
                    (positive_emb, False),
                    (negative_emb, True)
                ]:
                    for single_emb in emb:
                        prob = self.predict_binding(single_emb, reference_embeddings)
                        predicted_probs.append(prob)
                        true_labels.append(1 if is_binding else 0)
        
        true_labels = torch.tensor(true_labels, device=self.device)
        predicted_probs = torch.tensor(predicted_probs, device=self.device)
        
        gathered_true = [torch.zeros_like(true_labels) for _ in range(self.world_size)]
        gathered_probs = [torch.zeros_like(predicted_probs) for _ in range(self.world_size)]
        
        dist.all_gather(gathered_true, true_labels)
        dist.all_gather(gathered_probs, predicted_probs)
        
        if self.rank == 0:
            all_true_labels = torch.cat(gathered_true).cpu().numpy()
            all_predicted_probs = torch.cat(gathered_probs).cpu().numpy()
            predicted_labels = (all_predicted_probs >= 0.5).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(all_true_labels, predicted_labels),
                'precision': precision_score(all_true_labels, predicted_labels),
                'recall': recall_score(all_true_labels, predicted_labels),
                'roc_auc': roc_auc_score(all_true_labels, all_predicted_probs),
                'confusion_matrix': confusion_matrix(all_true_labels, predicted_labels).tolist()
            }
            
            self.logger.info("\nEvaluation Results:")
            self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"Precision: {metrics['precision']:.4f}")
            self.logger.info(f"Recall: {metrics['recall']:.4f}")
            self.logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            self.logger.info("\nConfusion Matrix:")
            self.logger.info(f"{metrics['confusion_matrix']}")
            
            # Save metrics
            with open('evaluation_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
            return metrics
        
        dist.barrier()
        return None

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
    logger.info(f"[Rank {rank}] Model initialized")

    trainer = TripletTrainer(model, train_dataset, val_dataset, test_dataset, rank, world_size, graph_metadata)

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
                torch.save(model.state_dict(), 'best_triplet_model.pth')
                logger.info(f'New best model saved at epoch {epoch}')

        trainer.scheduler.step()

        dist.barrier()

    if rank == 0:
        logger.info("Starting evaluation...")
    
    # Need to unwrap the model from DDP before evaluation
    model = trainer.model.module
    model.load_state_dict(torch.load('best_triplet_model.pth'))
    
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
    df = pd.read_parquet('cleaned_train_unique.parquet')
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    with open('unique_atom_and_edge_types.json', 'r') as f:
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
