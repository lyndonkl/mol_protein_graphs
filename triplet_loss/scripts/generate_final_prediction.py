"""
Memory-efficient implementation of protein binding predictions using FAISS-based nearest neighbor search.

This module provides functionality for generating protein binding predictions using FAISS-based
nearest neighbor search with memory optimization techniques.
"""

# Standard library imports
import gc
import json
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Dict

# Third-party imports
import numpy as np
import pandas as pd
import psutil
import faiss
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryTracker:
    """Utility class for tracking memory usage during processing."""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in GB."""
        return psutil.Process().memory_info().rss / 1024 / 1024 / 1024
    
    @staticmethod
    def get_available_memory() -> float:
        """Get available system memory in GB."""
        return psutil.virtual_memory().available / 1024 / 1024 / 1024
    
    @staticmethod
    def log_memory_usage(context: str):
        """Log current memory usage with context."""
        usage = MemoryTracker.get_memory_usage()
        available = MemoryTracker.get_available_memory()
        logger.info(f"Memory [{context}] - Used: {usage:.2f}GB, Available: {available:.2f}GB")

class PredictionGenerator:
    """
    Main class for generating protein binding predictions using FAISS-based nearest neighbor search.
    """
    
    def __init__(self, 
                 reference_dir: str = 'reference_embeddings',
                 test_dir: str = 'test_embeddings',
                 output_dir: str = 'submissions',
                 k_neighbors: int = 5,
                 memory_threshold: float = 0.8,
                 n_jobs: int = -1):
        """
        Initialize the PredictionGenerator.
        
        Args:
            reference_dir: Directory containing reference embeddings
            test_dir: Directory containing test embeddings
            output_dir: Directory for output predictions
            k_neighbors: Number of nearest neighbors to use
            memory_threshold: Memory usage threshold for garbage collection
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.reference_dir = reference_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.k_neighbors = k_neighbors
        self.memory_threshold = memory_threshold
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        
        # Load protein types from manifest
        manifest_path = os.path.join(reference_dir, 'manifest.json')
        with open(manifest_path, 'r') as f:
            self.protein_types = json.load(f)['protein_types']
        
        self.indices = {}  # Store FAISS indices for each protein
        
        logger.info(f"Initialized PredictionGenerator with {len(self.protein_types)} protein types")
    
    def _load_reference_data(self, protein_id: str) -> Dict[str, np.ndarray]:
        """Load reference data with memory optimization."""
        ref_path = os.path.join(self.reference_dir, f'reference_embeddings_protein_{protein_id}.pkl')
        
        try:
            with open(ref_path, 'rb') as f:
                data = pickle.load(f)
            
            binding_embeddings = np.array(data['binding'], dtype=np.float32)
            non_binding_embeddings = np.array(data['non_binding'], dtype=np.float32)
            
            embeddings = np.vstack([binding_embeddings, non_binding_embeddings])
            labels = np.concatenate([
                np.ones(len(binding_embeddings), dtype=np.int8),
                np.zeros(len(non_binding_embeddings), dtype=np.int8)
            ])
            
            logger.info(f"Loaded {len(binding_embeddings)} binding and {len(non_binding_embeddings)} "
                       f"non-binding embeddings for protein {protein_id}")
            
            return {'embeddings': embeddings, 'labels': labels}
            
        except MemoryError:
            logger.warning(f"Memory mapping fallback for protein {protein_id}")
            return np.load(ref_path, mmap_mode='r')
    
    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build an optimized FAISS index."""
        dimension = embeddings.shape[1]
        
        # Use IVFFlat index with inner product distance
        nlist = min(4096, max(4, int(np.sqrt(len(embeddings)))))  # Number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        
        # Train the index
        index.train(embeddings)
        index.add(embeddings)
        
        # Set number of probes for search
        index.nprobe = min(64, nlist)
        
        return index

    def _process_protein(self, protein_id: str):
        """Process a single protein type."""
        try:
            MemoryTracker.log_memory_usage(f"Before processing protein {protein_id}")
            
            if protein_id in self.indices:
                del self.indices[protein_id]
                gc.collect()
            
            ref_data = self._load_reference_data(protein_id)
            index = self._build_faiss_index(ref_data['embeddings'])
            
            self.indices[protein_id] = {
                'index': index,
                'labels': ref_data['labels']
            }
            
            logger.info(f"Built FAISS index for protein {protein_id}")
            
            del ref_data
            gc.collect()
            
            MemoryTracker.log_memory_usage(f"After processing protein {protein_id}")
            
        except Exception as e:
            logger.error(f"Error processing protein {protein_id}: {str(e)}")

    def build_indices(self):
        """Build FAISS indices for each protein type."""
        logger.info("Building FAISS indices for each protein type...")
        
        for protein_id in tqdm(self.protein_types):
            try:
                MemoryTracker.log_memory_usage(f"Before processing protein {protein_id}")
                
                # Clear previous protein data
                if protein_id in self.indices:
                    del self.indices[protein_id]
                    gc.collect()
                
                # Load data
                ref_data = self._load_reference_data(protein_id)
                
                # Build FAISS index
                dimension = ref_data['embeddings'].shape[1]
                index = faiss.IndexFlatL2(dimension)  # L2 distance
                index.add(ref_data['embeddings'].astype(np.float32))
                
                # Store index and labels
                self.indices[protein_id] = {
                    'index': index,
                    'labels': ref_data['labels']
                }
                
                logger.info(f"Built FAISS index for protein {protein_id}")
                
                del ref_data
                gc.collect()
                
                MemoryTracker.log_memory_usage(f"After processing protein {protein_id}")
                
            except Exception as e:
                logger.error(f"Error processing protein {protein_id}: {str(e)}")
                continue

    def predict_binding(self, query_embeddings: np.ndarray, protein_id: str) -> np.ndarray:
        """Predict binding using FAISS nearest neighbor search."""
        if protein_id not in self.indices:
            raise KeyError(f"No index found for protein {protein_id}. Did you run build_indices()?")
            
        index_data = self.indices[protein_id]
        
        # Get k nearest neighbors
        _, neighbors = index_data['index'].search(query_embeddings.astype(np.float32), self.k_neighbors)
        
        nearest_labels = index_data['labels'][neighbors]
        predictions = (nearest_labels.mean(axis=1) > 0.5).astype(np.int8)
        
        return predictions
    
    def generate_predictions(self):
        """Generate predictions with batch processing."""
        logger.info("Loading test embeddings...")
        
        # Define batch size
        BATCH_SIZE = 10000  # You can adjust this value based on your memory constraints
        predictions = []
        
        # Load test data
        with open(os.path.join(self.test_dir, 'test_embeddings.pkl'), 'rb') as f:
            test_data = pickle.load(f)
        
        test_data['embeddings'] = test_data['embeddings'].astype(np.float32)
        
        # Process by protein and batch
        for protein_id in tqdm(self.protein_types):
            indices = np.where(test_data['protein_ids'] == protein_id)[0]
            
            for i in range(0, len(indices), BATCH_SIZE):
                if MemoryTracker.get_available_memory() < self.memory_threshold:
                    gc.collect()
                
                batch_indices = indices[i:i + BATCH_SIZE]
                batch_embeddings = test_data['embeddings'][batch_indices]
                batch_molecule_ids = test_data['molecule_ids'][batch_indices]
                
                batch_predictions = self.predict_binding(batch_embeddings, protein_id)
                
                # Create dictionary entries with correct column names
                for mol_id, pred in zip(batch_molecule_ids, batch_predictions):
                    predictions.append({
                        'id': int(mol_id),
                        'binds': int(pred)
                    })
                
                logger.info(f"Processed batch for protein {protein_id}: "
                           f"{i+1}-{min(i+BATCH_SIZE, len(indices))} of {len(indices)}")
        
        # Create DataFrame and save
        os.makedirs(self.output_dir, exist_ok=True)
        df = pd.DataFrame(predictions)
        
        if 'id' in df.columns:
            df.sort_values('id', inplace=True)
        else:
            logger.warning("Column 'id' not found in DataFrame. Columns present:", df.columns)
        
        output_path = os.path.join(self.output_dir, 'submission.csv')
        df.to_csv(output_path, index=False, compression='gzip')
        
        logger.info(f"Saved predictions for {len(df)} molecules to {output_path}")


if __name__ == "__main__":
    try:
        generator = PredictionGenerator(
            reference_dir='reference_embeddings',
            test_dir='test_embeddings',
            output_dir='submissions',
            k_neighbors=5,
            memory_threshold=0.8,
            n_jobs=-1
        )
        
        # Build indices first!
        generator.build_indices()
        
        # Then generate predictions
        generator.generate_predictions()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
