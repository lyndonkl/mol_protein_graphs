# Standard library imports
import os
import json

# Third-party imports
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import traceback
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import pickle

# Custom imports
from triplet_loss.src.utils.helpers import setup_logger, collate_fn

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
