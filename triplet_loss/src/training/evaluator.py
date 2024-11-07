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
from triplet_loss.src.utils.helpers import setup_logger, collate_fn, collate_fn_predict

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
            'shuffle': False
        }
        
        self.val_loader = DataLoader(val_dataset, sampler=self.val_sampler, collate_fn=collate_fn, **loader_kwargs)
        self.test_loader = DataLoader(test_dataset, sampler=self.test_sampler, collate_fn=collate_fn_predict, **loader_kwargs)

    def generate_reference_embeddings(self):
        """Generate and save reference embeddings iteratively."""
        if self.rank == 0:
            self.logger.info("Generating reference embeddings from validation set...")
            os.makedirs('reference_embeddings/temp', exist_ok=True)
        
        self.model.eval()
        binding_embeddings_by_protein = {}
        non_binding_embeddings_by_protein = {}
        
        # Keep track of saved parts for each protein
        saved_parts = {}
        
        with torch.no_grad():
            for batch_idx, (combined_data, combined_protein_id, _, num_nodes) in enumerate(
                tqdm(self.val_loader, desc="Generating embeddings", disable=(self.rank != 0))
            ):
                # Process batch
                combined_data = combined_data.to(self.device)
                combined_protein_id = combined_protein_id.to(self.device)
                outputs = self.model(combined_data, combined_protein_id)

                # Split triplets
                anchor_emb, positive_emb, negative_emb = torch.split(outputs, num_nodes)
                anchor_protein_id, positive_protein_id, negative_protein_id = torch.split(
                    combined_protein_id, num_nodes
                )
                
                # Process each protein type in batch
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
                
                # Periodically gather and save embeddings
                if (batch_idx + 1) % 200 == 0:
                    for prot_id in binding_embeddings_by_protein:
                        if binding_embeddings_by_protein[prot_id]:
                            try:
                                # Each rank calculates its local sizes
                                binding_size = torch.tensor([t.size(0) for t in binding_embeddings_by_protein[prot_id]]).sum()
                                non_binding_size = torch.tensor([t.size(0) for t in non_binding_embeddings_by_protein[prot_id]]).sum()
                                
                                # Share size with rank 0
                                if self.rank == 0:
                                    all_binding_sizes = [binding_size.item()]
                                    all_non_binding_sizes = [non_binding_size.item()]
                                    for i in range(1, self.world_size):
                                        size_tensor = torch.zeros_like(binding_size)
                                        dist.recv(size_tensor, src=i)
                                        all_binding_sizes.append(size_tensor.item())
                                        
                                        size_tensor = torch.zeros_like(non_binding_size)
                                        dist.recv(size_tensor, src=i)
                                        all_non_binding_sizes.append(size_tensor.item())
                                else:
                                    dist.send(binding_size, dst=0)
                                    dist.send(non_binding_size, dst=0)
                                
                                # Rank 0 checks sizes and broadcasts decision
                                proceed = torch.zeros(1, dtype=torch.bool)
                                if self.rank == 0:
                                    # No need to convert to integers since we already did above
                                    sizes_match = (len(set(all_binding_sizes)) == 1 and 
                                                 len(set(all_non_binding_sizes)) == 1)
                                    proceed[0] = sizes_match
                                    if not sizes_match:
                                        self.logger.warning(f"Size mismatch at batch {batch_idx} for protein {prot_id}: "
                                                         f"binding={all_binding_sizes}, "
                                                         f"non_binding={all_non_binding_sizes}")
                                
                                # Broadcast decision to all ranks
                                dist.broadcast(proceed, src=0)
                                
                                if proceed[0]:
                                    binding_emb = torch.cat(binding_embeddings_by_protein[prot_id])
                                    non_binding_emb = torch.cat(non_binding_embeddings_by_protein[prot_id])
                                    
                                    gathered_binding = [torch.zeros_like(binding_emb) for _ in range(self.world_size)]
                                    gathered_non_binding = [torch.zeros_like(non_binding_emb) for _ in range(self.world_size)]
                                    
                                    dist.barrier()
                                    dist.all_gather(gathered_binding, binding_emb)
                                    dist.all_gather(gathered_non_binding, non_binding_emb)
                                    dist.barrier()
                                    
                                    if self.rank == 0:
                                        reference_data = {
                                            'binding': torch.cat(gathered_binding).cpu().numpy(),
                                            'non_binding': torch.cat(gathered_non_binding).cpu().numpy()
                                        }
                                        
                                        part_filename = f'reference_embeddings_protein_{prot_id}_part_{batch_idx}.pkl'
                                        temp_path = os.path.join('reference_embeddings/temp', part_filename)
                                        with open(temp_path, 'wb') as f:
                                            pickle.dump(reference_data, f)
                                        
                                        if prot_id not in saved_parts:
                                            saved_parts[prot_id] = []
                                        saved_parts[prot_id].append(temp_path)
                                    
                                    # Clean up
                                    del gathered_binding, gathered_non_binding
                                    del binding_emb, non_binding_emb
                                
                            except Exception as e:
                                self.logger.error(f"[Rank {self.rank}] Failed to process protein {prot_id} "
                                                f"at batch {batch_idx}: {str(e)}")
                                continue
                    
                    binding_embeddings_by_protein = {}
                    non_binding_embeddings_by_protein = {}
                    dist.barrier()
        
        # Process any remaining embeddings at the end
        for prot_id in binding_embeddings_by_protein:
            if binding_embeddings_by_protein[prot_id]:
                try:
                    # Each rank calculates its local sizes
                    binding_size = torch.tensor([t.size(0) for t in binding_embeddings_by_protein[prot_id]]).sum()
                    non_binding_size = torch.tensor([t.size(0) for t in non_binding_embeddings_by_protein[prot_id]]).sum()
                    
                    # Share size with rank 0
                    if self.rank == 0:
                        all_binding_sizes = [binding_size.item()]
                        all_non_binding_sizes = [non_binding_size.item()]
                        for i in range(1, self.world_size):
                            size_tensor = torch.zeros_like(binding_size)
                            dist.recv(size_tensor, src=i)
                            all_binding_sizes.append(size_tensor.item())
                            
                            size_tensor = torch.zeros_like(non_binding_size)
                            dist.recv(size_tensor, src=i)
                            all_non_binding_sizes.append(size_tensor.item())
                    else:
                        dist.send(binding_size, dst=0)
                        dist.send(non_binding_size, dst=0)
                    
                    # Rank 0 checks sizes and broadcasts decision
                    proceed = torch.zeros(1, dtype=torch.bool)
                    if self.rank == 0:
                        sizes_match = (len(set(all_binding_sizes)) == 1 and 
                                     len(set(all_non_binding_sizes)) == 1)
                        proceed[0] = sizes_match
                        if not sizes_match:
                            self.logger.warning(f"Size mismatch for final protein {prot_id}: "
                                             f"binding={all_binding_sizes}, "
                                             f"non_binding={all_non_binding_sizes}")
                    
                    # Broadcast decision to all ranks
                    dist.broadcast(proceed, src=0)
                    
                    if proceed[0]:
                        binding_emb = torch.cat(binding_embeddings_by_protein[prot_id])
                        non_binding_emb = torch.cat(non_binding_embeddings_by_protein[prot_id])
                        
                        gathered_binding = [torch.zeros_like(binding_emb) for _ in range(self.world_size)]
                        gathered_non_binding = [torch.zeros_like(non_binding_emb) for _ in range(self.world_size)]
                        
                        dist.barrier()
                        dist.all_gather(gathered_binding, binding_emb)
                        dist.all_gather(gathered_non_binding, non_binding_emb)
                        dist.barrier()
                        
                        if self.rank == 0:
                            reference_data = {
                                'binding': torch.cat(gathered_binding).cpu().numpy(),
                                'non_binding': torch.cat(gathered_non_binding).cpu().numpy()
                            }
                            
                            part_filename = f'reference_embeddings_protein_{prot_id}_part_final.pkl'
                            temp_path = os.path.join('reference_embeddings/temp', part_filename)
                            with open(temp_path, 'wb') as f:
                                pickle.dump(reference_data, f)
                            
                            if prot_id not in saved_parts:
                                saved_parts[prot_id] = []
                            saved_parts[prot_id].append(temp_path)
                        
                        # Clean up
                        del gathered_binding, gathered_non_binding
                        del binding_emb, non_binding_emb
                    
                except Exception as e:
                    self.logger.error(f"[Rank {self.rank}] Failed to process final protein {prot_id}: {str(e)}")
                    continue

        binding_embeddings_by_protein = {}
        non_binding_embeddings_by_protein = {}
        dist.barrier()
        
        # Combine all parts into final files
        if self.rank == 0:
            self.logger.info("Combining partial embeddings into final files...")
            for prot_id in saved_parts:
                all_binding = []
                all_non_binding = []
                
                # Load and combine all parts
                for part_path in saved_parts[prot_id]:
                    with open(part_path, 'rb') as f:
                        part_data = pickle.load(f)
                        all_binding.append(part_data['binding'])
                        all_non_binding.append(part_data['non_binding'])
                
                # Save combined data
                final_data = {
                    'binding': np.concatenate(all_binding),
                    'non_binding': np.concatenate(all_non_binding)
                }
                
                final_path = os.path.join('reference_embeddings', f'reference_embeddings_protein_{prot_id}.pkl')
                with open(final_path, 'wb') as f:
                    pickle.dump(final_data, f)
                
                # Clean up temp files
                for part_path in saved_parts[prot_id]:
                    os.remove(part_path)
            
            # Save manifest
            manifest = {
                'protein_types': list(saved_parts.keys()),
                'timestamp': datetime.datetime.now().isoformat()
            }
            with open(os.path.join('reference_embeddings', 'manifest.json'), 'w') as f:
                json.dump(manifest, f, indent=4)
            
            self.logger.info("Reference embeddings generation complete!")

            # Remove temp directory
            os.rmdir('reference_embeddings/temp')
        
        dist.barrier()  # Final sync

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
        Generate embeddings for test dataset and save them periodically.
        
        Returns:
            None
        """
        if self.rank == 0:
            self.logger.info("Starting test embeddings generation...")
        
        # Check for reference embeddings and generate if missing
        if not os.path.exists('reference_embeddings/manifest.json'):
            self.generate_reference_embeddings()
            dist.barrier()
        
        if self.rank == 0:
            os.makedirs('test_embeddings/temp', exist_ok=True)
        
        self.model.eval()
        current_embeddings = []
        current_mol_ids = []
        current_protein_ids = []
        saved_parts = []
        
        with torch.no_grad():
            for batch_idx, (node, protein_type_id, _, _) in enumerate(
                tqdm(self.test_loader, desc="Generating embeddings", disable=(self.rank != 0))
            ):
                node = node.to(self.device)
                protein_type_id = protein_type_id.to(self.device)
                batch_emb = self.model(node, protein_type_id)
                
                # Get molecule IDs and protein IDs
                mol_ids = [node['smolecule'].id[i] for i in range(len(node['smolecule'].id))]
                protein_ids = protein_type_id.cpu().tolist()
                
                # Store embeddings, IDs, and protein types
                current_embeddings.extend(batch_emb.cpu().numpy())
                current_mol_ids.extend(mol_ids)
                current_protein_ids.extend(protein_ids)
                
                # Save every 200 batches
                if (batch_idx + 1) % 200 == 0:
                    try:
                        # Each rank calculates its local sizes
                        embeddings_size = torch.tensor([len(current_embeddings)])
                        mol_ids_size = torch.tensor([len(current_mol_ids)])
                        protein_ids_size = torch.tensor([len(current_protein_ids)])
                        
                        # Share sizes with rank 0
                        if self.rank == 0:
                            all_embeddings_sizes = [embeddings_size.item()]
                            all_mol_ids_sizes = [mol_ids_size.item()]
                            all_protein_ids_sizes = [protein_ids_size.item()]
                            for i in range(1, self.world_size):
                                size_tensor = torch.zeros_like(embeddings_size)
                                dist.recv(size_tensor, src=i)
                                all_embeddings_sizes.append(size_tensor.item())
                                
                                size_tensor = torch.zeros_like(mol_ids_size)
                                dist.recv(size_tensor, src=i)
                                all_mol_ids_sizes.append(size_tensor.item())
                                
                                size_tensor = torch.zeros_like(protein_ids_size)
                                dist.recv(size_tensor, src=i)
                                all_protein_ids_sizes.append(size_tensor.item())
                        else:
                            dist.send(embeddings_size, dst=0)
                            dist.send(mol_ids_size, dst=0)
                            dist.send(protein_ids_size, dst=0)
                        
                        # Rank 0 checks sizes and broadcasts decision
                        proceed = torch.zeros(1, dtype=torch.bool)
                        if self.rank == 0:
                            sizes_match = (len(set(all_embeddings_sizes)) == 1 and 
                                         len(set(all_mol_ids_sizes)) == 1 and
                                         len(set(all_protein_ids_sizes)) == 1 and
                                         all_embeddings_sizes[0] == all_mol_ids_sizes[0] == all_protein_ids_sizes[0])
                            proceed[0] = sizes_match
                            if not sizes_match:
                                self.logger.warning(f"Size mismatch at batch {batch_idx}: "
                                                 f"embeddings={all_embeddings_sizes}, "
                                                 f"mol_ids={all_mol_ids_sizes}, "
                                                 f"protein_ids={all_protein_ids_sizes}")
                        
                        # Broadcast decision to all ranks
                        dist.broadcast(proceed, src=0)
                        
                        if proceed[0]:
                            embeddings = torch.tensor(current_embeddings, device=self.device)
                            molecule_ids = torch.tensor([int(mid) for mid in current_mol_ids], device=self.device)
                            protein_ids = torch.tensor(current_protein_ids, device=self.device)
                            
                            gathered_embeddings = [torch.zeros_like(embeddings) for _ in range(self.world_size)]
                            gathered_ids = [torch.zeros_like(molecule_ids) for _ in range(self.world_size)]
                            gathered_protein_ids = [torch.zeros_like(protein_ids) for _ in range(self.world_size)]
                            
                            dist.barrier()
                            dist.all_gather(gathered_embeddings, embeddings)
                            dist.all_gather(gathered_ids, molecule_ids)
                            dist.all_gather(gathered_protein_ids, protein_ids)
                            dist.barrier()
                            
                            if self.rank == 0:
                                all_embeddings = torch.cat(gathered_embeddings).cpu().numpy()
                                all_molecule_ids = torch.cat(gathered_ids).cpu().numpy()
                                all_protein_ids = torch.cat(gathered_protein_ids).cpu().numpy()
                                
                                # Save part
                                part_filename = f'test_embeddings_part_{batch_idx}.pkl'
                                temp_path = os.path.join('test_embeddings/temp', part_filename)
                                with open(temp_path, 'wb') as f:
                                    pickle.dump({
                                        'embeddings': all_embeddings,
                                        'molecule_ids': all_molecule_ids,
                                        'protein_ids': all_protein_ids
                                    }, f)
                                saved_parts.append(temp_path)
                            
                            # Clean up
                            del gathered_embeddings, gathered_ids, gathered_protein_ids
                            del embeddings, molecule_ids, protein_ids
                        
                        # Clear current batch data
                        current_embeddings = []
                        current_mol_ids = []
                        current_protein_ids = []
                        
                    except Exception as e:
                        self.logger.error(f"[Rank {self.rank}] Failed to process batch {batch_idx}: {str(e)}")
                        continue
                    
                    dist.barrier()
        
        # Process remaining embeddings
        if current_embeddings:
            try:
                # Each rank calculates its local sizes
                embeddings_size = torch.tensor([len(current_embeddings)])
                mol_ids_size = torch.tensor([len(current_mol_ids)])
                protein_ids_size = torch.tensor([len(current_protein_ids)])
                
                # Share sizes with rank 0
                if self.rank == 0:
                    all_embeddings_sizes = [embeddings_size.item()]
                    all_mol_ids_sizes = [mol_ids_size.item()]
                    all_protein_ids_sizes = [protein_ids_size.item()]
                    for i in range(1, self.world_size):
                        size_tensor = torch.zeros_like(embeddings_size)
                        dist.recv(size_tensor, src=i)
                        all_embeddings_sizes.append(size_tensor.item())
                        
                        size_tensor = torch.zeros_like(mol_ids_size)
                        dist.recv(size_tensor, src=i)
                        all_mol_ids_sizes.append(size_tensor.item())
                        
                        size_tensor = torch.zeros_like(protein_ids_size)
                        dist.recv(size_tensor, src=i)
                        all_protein_ids_sizes.append(size_tensor.item())
                else:
                    dist.send(embeddings_size, dst=0)
                    dist.send(mol_ids_size, dst=0)
                    dist.send(protein_ids_size, dst=0)
                
                # Rank 0 checks sizes and broadcasts decision
                proceed = torch.zeros(1, dtype=torch.bool)
                if self.rank == 0:
                    sizes_match = (len(set(all_embeddings_sizes)) == 1 and 
                                 len(set(all_mol_ids_sizes)) == 1 and
                                 len(set(all_protein_ids_sizes)) == 1 and
                                 all_embeddings_sizes[0] == all_mol_ids_sizes[0] == all_protein_ids_sizes[0])
                    proceed[0] = sizes_match
                    if not sizes_match:
                        self.logger.warning(f"Size mismatch for final batch: "
                                         f"embeddings={all_embeddings_sizes}, "
                                         f"mol_ids={all_mol_ids_sizes}, "
                                         f"protein_ids={all_protein_ids_sizes}")
                
                # Broadcast decision to all ranks
                dist.broadcast(proceed, src=0)
                
                if proceed[0]:
                    embeddings = torch.tensor(current_embeddings, device=self.device)
                    molecule_ids = torch.tensor([int(mid) for mid in current_mol_ids], device=self.device)
                    protein_ids = torch.tensor(current_protein_ids, device=self.device)
                    
                    gathered_embeddings = [torch.zeros_like(embeddings) for _ in range(self.world_size)]
                    gathered_ids = [torch.zeros_like(molecule_ids) for _ in range(self.world_size)]
                    gathered_protein_ids = [torch.zeros_like(protein_ids) for _ in range(self.world_size)]
                    
                    dist.barrier()
                    dist.all_gather(gathered_embeddings, embeddings)
                    dist.all_gather(gathered_ids, molecule_ids)
                    dist.all_gather(gathered_protein_ids, protein_ids)
                    dist.barrier()
                    
                    if self.rank == 0:
                        all_embeddings = torch.cat(gathered_embeddings).cpu().numpy()
                        all_molecule_ids = torch.cat(gathered_ids).cpu().numpy()
                        all_protein_ids = torch.cat(gathered_protein_ids).cpu().numpy()
                        
                        part_filename = 'test_embeddings_part_final.pkl'
                        temp_path = os.path.join('test_embeddings/temp', part_filename)
                        with open(temp_path, 'wb') as f:
                            pickle.dump({
                                'embeddings': all_embeddings,
                                'molecule_ids': all_molecule_ids,
                                'protein_ids': all_protein_ids
                            }, f)
                        saved_parts.append(temp_path)
                    
                    # Clean up
                    del gathered_embeddings, gathered_ids, gathered_protein_ids
                    del embeddings, molecule_ids, protein_ids
            
            except Exception as e:
                self.logger.error(f"[Rank {self.rank}] Failed to process final batch: {str(e)}")
        
        dist.barrier()
        
        # Combine all parts into final file
        if self.rank == 0:
            self.logger.info("Combining partial embeddings into final file...")
            all_embeddings = []
            all_molecule_ids = []
            all_protein_ids = []
            
            # Load and combine all parts
            for part_path in saved_parts:
                with open(part_path, 'rb') as f:
                    part_data = pickle.load(f)
                    all_embeddings.append(part_data['embeddings'])
                    all_molecule_ids.append(part_data['molecule_ids'])
                    all_protein_ids.append(part_data['protein_ids'])
            
            # Save combined data
            final_data = {
                'embeddings': np.concatenate(all_embeddings),
                'molecule_ids': np.concatenate(all_molecule_ids),
                'protein_ids': np.concatenate(all_protein_ids),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            os.makedirs('test_embeddings', exist_ok=True)
            final_path = os.path.join('test_embeddings', 'test_embeddings.pkl')
            with open(final_path, 'wb') as f:
                pickle.dump(final_data, f)
            
            # Clean up temp files
            for part_path in saved_parts:
                os.remove(part_path)
            
            self.logger.info(f"Test embeddings saved to {final_path}")
            os.rmdir('test_embeddings/temp')
        
        dist.barrier()
