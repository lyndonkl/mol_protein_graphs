"""
Evaluator module for triplet loss model that handles distributed evaluation
and generates binding predictions for molecules.
"""

# Standard library imports
import os
import warnings
import json
import datetime
import pickle
import traceback

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Custom imports
from triplet_loss.src.utils.helpers import setup_logger, collate_fn

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


class TripletEvaluator:
    """
    Evaluator class for distributed evaluation of triplet loss models.
    Handles reference embedding generation and binding predictions.
    """
    
    def __init__(self, model, val_dataset, test_dataset, rank, world_size, batch_size=32, device='cuda'):
        """
        Initialize the evaluator with model and datasets.
        
        Args:
            model: The neural network model
            val_dataset: Validation dataset for generating reference embeddings
            test_dataset: Test dataset for predictions
            rank: Process rank in distributed setting
            world_size: Total number of processes
            batch_size: Batch size for data loading
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.logger = setup_logger()
        
        # Initialize distributed model
        try:
            self.model = DistributedDataParallel(
                self.model, 
                device_ids=None, 
                find_unused_parameters=True
            )
            self.logger.info(f"[Rank {rank}] Evaluation model wrapped with DistributedDataParallel")
        except Exception as e:
            self.logger.error(f"[Rank {rank}] Exception during model wrapping: {e}")
            traceback.print_exc()
            raise e

        # Setup data loaders with distributed sampling
        self.val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        self.test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': 2,
            'shuffle': False,
            'collate_fn': collate_fn
        }
        
        self.val_loader = DataLoader(val_dataset, sampler=self.val_sampler, **loader_kwargs)
        self.test_loader = DataLoader(test_dataset, sampler=self.test_sampler, **loader_kwargs)
        
        self.reference_path = 'reference_embeddings.pkl'

    def generate_reference_embeddings(self):
        """Generate and save reference embeddings from validation set."""
        if self.rank == 0:
            self.logger.info("Generating reference embeddings from validation set...")
            
        self.model.eval()
        binding_embeddings_by_protein = {}
        non_binding_embeddings_by_protein = {}
        
        with torch.no_grad():
            for combined_data, combined_protein_id, _, num_nodes in tqdm(
                self.val_loader,
                desc="Generating embeddings",
                disable=(self.rank != 0)
            ):
                # Process batch
                combined_data = combined_data.to(self.device)
                combined_protein_id = combined_protein_id.to(self.device)
                outputs = self.model(combined_data, combined_protein_id)

                # Split triplets
                anchor_emb, positive_emb, negative_emb = torch.split(outputs, num_nodes)
                anchor_protein_id, positive_protein_id, negative_protein_id = torch.split(
                    combined_protein_id, 
                    num_nodes
                )
                
                # Process each protein type
                for prot_type in combined_protein_id.unique():
                    prot_id = prot_type.item()
                    
                    # Initialize storage for new protein types
                    if prot_id not in binding_embeddings_by_protein:
                        binding_embeddings_by_protein[prot_id] = []
                        non_binding_embeddings_by_protein[prot_id] = []
                    
                    # Store embeddings by type
                    for mask, emb, is_binding in [
                        (anchor_protein_id == prot_type, anchor_emb, False),
                        (positive_protein_id == prot_type, positive_emb, False),
                        (negative_protein_id == prot_type, negative_emb, True)
                    ]:
                        if mask.any():
                            target_list = (binding_embeddings_by_protein if is_binding 
                                         else non_binding_embeddings_by_protein)[prot_id]
                            target_list.append(emb[mask])
        
        # Concatenate embeddings
        for prot_id in binding_embeddings_by_protein:
            binding_embeddings_by_protein[prot_id] = torch.cat(binding_embeddings_by_protein[prot_id])
            non_binding_embeddings_by_protein[prot_id] = torch.cat(non_binding_embeddings_by_protein[prot_id])
        
        # Gather embeddings from all processes
        gathered_binding = {
            prot_id: [torch.zeros_like(emb) for _ in range(self.world_size)]
            for prot_id, emb in binding_embeddings_by_protein.items()
        }
        gathered_non_binding = {
            prot_id: [torch.zeros_like(emb) for _ in range(self.world_size)]
            for prot_id, emb in non_binding_embeddings_by_protein.items()
        }
        
        for prot_id in binding_embeddings_by_protein:
            dist.all_gather(gathered_binding[prot_id], binding_embeddings_by_protein[prot_id])
            dist.all_gather(gathered_non_binding[prot_id], non_binding_embeddings_by_protein[prot_id])
        
        # Save reference embeddings (rank 0 only)
        if self.rank == 0:
            os.makedirs('reference_embeddings', exist_ok=True)
            
            for prot_id in binding_embeddings_by_protein:
                reference_data = {
                    'binding': torch.cat(gathered_binding[prot_id]).cpu().numpy(),
                    'non_binding': torch.cat(gathered_non_binding[prot_id]).cpu().numpy()
                }
                
                reference_path = os.path.join('reference_embeddings', f'reference_embeddings_protein_{prot_id}.pkl')
                with open(reference_path, 'wb') as f:
                    pickle.dump(reference_data, f)
            
            # Save manifest
            manifest = {
                'protein_types': list(binding_embeddings_by_protein.keys()),
                'timestamp': datetime.datetime.now().isoformat()
            }
            with open(os.path.join('reference_embeddings', 'manifest.json'), 'w') as f:
                json.dump(manifest, f, indent=4)
            
            self.logger.info("Reference embeddings generated and saved by protein type.")
        
        dist.barrier()

    def load_reference_embeddings(self):
        """Load reference embeddings for each protein type from saved files."""
        manifest_path = os.path.join('reference_embeddings', 'manifest.json')
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError("Reference embeddings manifest not found. Run generate_reference_embeddings() first.")
            
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        reference_embeddings = {}
        for prot_id in manifest['protein_types']:
            reference_path = os.path.join('reference_embeddings', f'reference_embeddings_protein_{prot_id}.pkl')
            with open(reference_path, 'rb') as f:
                reference_embeddings[prot_id] = pickle.load(f)
                
        return reference_embeddings

    def predict_binding_batch(self, query_embeddings, reference_embeddings, k=5):
        """
        Predict binding for a batch of molecule embeddings.
        
        Args:
            query_embeddings: Batch of embeddings to predict (N x embedding_dim)
            reference_embeddings: Dict containing 'binding' and 'non_binding' reference embeddings
            k: Number of nearest neighbors to consider
            
        Returns:
            numpy array: Array of predictions (0 or 1) for each query embedding
        """
        if torch.is_tensor(query_embeddings):
            query_embeddings = query_embeddings.cpu().numpy()
        
        # Calculate pairwise distances
        binding_distances = np.linalg.norm(
            query_embeddings[:, np.newaxis, :] - reference_embeddings['binding'][np.newaxis, :, :],
            axis=2
        )
        non_binding_distances = np.linalg.norm(
            query_embeddings[:, np.newaxis, :] - reference_embeddings['non_binding'][np.newaxis, :, :],
            axis=2
        )
        
        # Get k nearest neighbors
        k_nearest_binding = np.partition(binding_distances, k, axis=1)[:, :k]
        k_nearest_non_binding = np.partition(non_binding_distances, k, axis=1)[:, :k]
        
        # Calculate average distances and probabilities
        avg_binding_dist = np.mean(k_nearest_binding, axis=1)
        avg_non_binding_dist = np.mean(k_nearest_non_binding, axis=1)
        
        distances = np.stack([avg_non_binding_dist, avg_binding_dist], axis=1)
        probabilities = F.softmax(torch.tensor(-distances), dim=1).numpy()
        
        return (probabilities[:, 1] > probabilities[:, 0]).astype(int)

    def predict(self):
        """
        Generate binding predictions for test dataset.
        
        Returns:
            numpy array: Predictions if rank is 0, None otherwise
        """
        if self.rank == 0:
            self.logger.info("Starting evaluation...")
        
        if not os.path.exists(self.reference_path):
            self.generate_reference_embeddings()
        reference_embeddings = self.load_reference_embeddings()
        
        self.model.eval()
        all_predictions = []
        all_mol_ids = []
        all_embeddings = []
        
        with torch.no_grad():
            for node, protein_type_id in tqdm(
                self.test_loader,
                desc="Evaluating",
                disable=(self.rank != 0)
            ):
                node = node.to(self.device)
                protein_type_id = protein_type_id.to(self.device)
                batch_emb = self.model(node, protein_type_id)
                
                # Process predictions by protein type
                for prot_id in protein_type_id.unique():
                    mask = protein_type_id == prot_id
                    if mask.any():
                        prot_emb = batch_emb[mask]
                        preds = self.predict_binding_batch(prot_emb, reference_embeddings[prot_id.item()])
                        all_predictions.extend(preds)
                        all_embeddings.extend(prot_emb.cpu().numpy())
                        
                        mol_ids = [node[i]['smolecule'].id for i in range(len(node)) if mask[i]]
                        all_mol_ids.extend(mol_ids)
        
        # Gather predictions and embeddings from all processes
        predicted_probs = torch.tensor(all_predictions, device=self.device)
        molecule_ids = torch.tensor([int(mid) for mid in all_mol_ids], device=self.device)
        embeddings = torch.tensor(all_embeddings, device=self.device)
        
        gathered_probs = [torch.zeros_like(predicted_probs) for _ in range(self.world_size)]
        gathered_ids = [torch.zeros_like(molecule_ids) for _ in range(self.world_size)]
        gathered_embeddings = [torch.zeros_like(embeddings) for _ in range(self.world_size)]
        
        dist.all_gather(gathered_probs, predicted_probs)
        dist.all_gather(gathered_ids, molecule_ids)
        dist.all_gather(gathered_embeddings, embeddings)
        
        # Save results (rank 0 only)
        if self.rank == 0:
            all_predicted_probs = torch.cat(gathered_probs).cpu().numpy()
            all_molecule_ids = torch.cat(gathered_ids).cpu().numpy()
            all_embeddings = torch.cat(gathered_embeddings).cpu().numpy()
            
            # Save predictions
            results_df = pd.DataFrame({
                'id': all_molecule_ids,
                'binds': all_predicted_probs
            })
            
            os.makedirs('results', exist_ok=True)
            output_path = os.path.join('results', 'binding_predictions.csv')
            results_df.to_csv(output_path, index=False)
            self.logger.info(f"Predictions saved to {output_path}")
            
            # Save embeddings
            embeddings_df = pd.DataFrame({
                'id': all_molecule_ids,
                'embeddings': [emb.tolist() for emb in all_embeddings]
            })
            
            embeddings_path = os.path.join('results', 'test_molecule_embeddings.csv')
            embeddings_df.to_csv(embeddings_path, index=False)
            self.logger.info(f"Embeddings saved to {embeddings_path}")
        
        dist.barrier()
        return all_predicted_probs if self.rank == 0 else None
