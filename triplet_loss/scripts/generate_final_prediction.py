"""
Script to generate final binding predictions using protein-specific optimized 
multi-level hierarchical clustering.
"""

import os
import math
import time
import sys
import pickle
import json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class TreeOptimizer:
    @staticmethod
    def calculate_tree_parameters(total_embeddings, target_leaf_size=100):
        """Calculate optimal tree parameters based on dataset size."""
        N = total_embeddings
        L = target_leaf_size
        
        best_params = None
        min_error = float('inf')
        
        for b in range(2, 51):
            d = round(math.log(N/L, b))
            expected_leaves = b**d
            actual_leaf_size = N / expected_leaves
            error = abs(actual_leaf_size - L)
            
            if error < min_error:
                min_error = error
                best_params = (d, b)
        
        return best_params

    @staticmethod
    def get_leaf_sizes(hierarchy):
        """Recursively get sizes of all leaf nodes in the hierarchy."""
        if not hierarchy['children']:
            return [len(hierarchy['embeddings'])]
        
        leaf_sizes = []
        for child in hierarchy['children']:
            leaf_sizes.extend(TreeOptimizer.get_leaf_sizes(child))
        return leaf_sizes

    @staticmethod
    def benchmark_protein_parameters(embeddings, depths, branches, k_neighbors=5):
        """Benchmark different tree configurations for a specific protein dataset."""
        results = []
        
        for depth in depths:
            for branch in branches:
                # Create test hierarchy
                hierarchy = {
                    'embeddings': embeddings,
                    'labels': np.zeros(len(embeddings)),  # Dummy labels for testing
                    'children': [],
                    'center': np.mean(embeddings, axis=0)
                }
                
                # Time tree construction
                start_time = time.time()
                TreeOptimizer._build_test_hierarchy(hierarchy, depth=0, max_depth=depth, 
                                                  branching_factor=branch)
                build_time = time.time() - start_time
                
                # Get leaf statistics
                leaf_sizes = TreeOptimizer.get_leaf_sizes(hierarchy)
                
                results.append({
                    'depth': depth,
                    'branching_factor': branch,
                    'build_time': build_time,
                    'avg_leaf_size': np.mean(leaf_sizes),
                    'leaf_size_std': np.std(leaf_sizes),
                    'num_leaves': len(leaf_sizes)
                })
        
        return pd.DataFrame(results)

    @staticmethod
    def _build_test_hierarchy(node, depth, max_depth, branching_factor):
        """Build test hierarchy for benchmarking."""
        if depth >= max_depth or len(node['embeddings']) <= branching_factor:
            return
            
        linkage_matrix = linkage(node['embeddings'], method='ward')
        cluster_labels = fcluster(linkage_matrix, branching_factor, criterion='maxclust')
        
        for i in range(1, branching_factor + 1):
            mask = cluster_labels == i
            if not np.any(mask):
                continue
                
            child = {
                'embeddings': node['embeddings'][mask],
                'labels': node['labels'][mask],
                'children': [],
                'center': np.mean(node['embeddings'][mask], axis=0)
            }
            node['children'].append(child)
            
            TreeOptimizer._build_test_hierarchy(child, depth + 1, max_depth, branching_factor)

class PredictionGenerator:
    def __init__(self, reference_dir='reference_embeddings', test_dir='test_embeddings',
                 output_dir='submissions', k_neighbors=5):
        self.reference_dir = reference_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.k_neighbors = k_neighbors
        
        # Load protein types from manifest
        manifest_path = os.path.join(reference_dir, 'manifest.json')
        with open(manifest_path, 'r') as f:
            self.protein_types = json.load(f)['protein_types']
        
        # Store protein-specific parameters
        self.protein_parameters = {}
        self.reference_data = {}
    
    def _load_reference_data(self, protein_id):
        """Load and combine reference embeddings for a protein."""
        ref_path = os.path.join(self.reference_dir, f'reference_embeddings_protein_{protein_id}.pkl')
        with open(ref_path, 'rb') as f:
            ref_data = pickle.load(f)
        
        return {
            'embeddings': np.concatenate([ref_data['binding'], ref_data['non_binding']]),
            'labels': np.concatenate([
                np.ones(len(ref_data['binding'])),
                np.zeros(len(ref_data['non_binding']))
            ])
        }
    
    def optimize_protein_parameters(self):
        """Optimize tree parameters for each protein type."""
        print("Optimizing tree parameters for each protein type...")
        
        for protein_id in tqdm(self.protein_types):
            print(f"\nOptimizing parameters for protein {protein_id}")
            
            # Load protein data
            protein_data = self._load_reference_data(protein_id)
            num_embeddings = len(protein_data['embeddings'])
            
            # Calculate theoretical optimal parameters
            depth, branch = TreeOptimizer.calculate_tree_parameters(num_embeddings)

            self.protein_parameters[protein_id] = {
                'max_depth': depth,
                'branching_factor': branch
            }
            
            print(f"Selected parameters for protein {protein_id}:")
            print(f"  depth={self.protein_parameters[protein_id]['max_depth']}")
            print(f"  branching_factor={self.protein_parameters[protein_id]['branching_factor']}")

    def generate_multilevel_clusters(self):
        """Generate multi-level hierarchical clustering for each protein type."""
        print("Generating multi-level hierarchical clusters...")
        
        # Optimize parameters if not already done
        if not self.protein_parameters:
            self.optimize_protein_parameters()
        
        for protein_id in tqdm(self.protein_types):
            ref_data = self._load_reference_data(protein_id)
            
            # Build hierarchical tree structure
            hierarchy = {
                'embeddings': ref_data['embeddings'],
                'labels': ref_data['labels'],
                'children': [],
                'center': np.mean(ref_data['embeddings'], axis=0)
            }
            
            self._build_hierarchy(hierarchy, depth=0, protein_id=protein_id)
            self.reference_data[protein_id] = hierarchy

    def _build_hierarchy(self, node, depth, protein_id):
        """Recursively build hierarchy of clusters using protein-specific parameters."""
        params = self.protein_parameters[protein_id]
        
        if depth >= params['max_depth'] or len(node['embeddings']) <= params['branching_factor']:
            return
            
        linkage_matrix = linkage(node['embeddings'], method='ward')
        cluster_labels = fcluster(linkage_matrix, params['branching_factor'], criterion='maxclust')
        
        for i in range(1, params['branching_factor'] + 1):
            mask = cluster_labels == i
            if not np.any(mask):
                continue
                
            child = {
                'embeddings': node['embeddings'][mask],
                'labels': node['labels'][mask],
                'children': [],
                'center': np.mean(node['embeddings'][mask], axis=0)
            }
            node['children'].append(child)
            
            self._build_hierarchy(child, depth + 1, protein_id)

    def predict_binding(self, query_embeddings, protein_id):
        """Use multi-level hierarchy for efficient prediction."""
        hierarchy = self.reference_data[protein_id]
        predictions = []
        
        for query in query_embeddings:
            # Navigate down the hierarchy
            current_node = hierarchy
            while current_node['children']:
                # Find nearest child cluster
                distances = cdist([query], 
                                [child['center'] for child in current_node['children']])[0]
                nearest_idx = np.argmin(distances)
                current_node = current_node['children'][nearest_idx]
            
            # At leaf node, find k nearest neighbors
            distances = cdist([query], current_node['embeddings'])[0]
            nearest_indices = np.argsort(distances)[:self.k_neighbors]
            nearest_labels = current_node['labels'][nearest_indices]
            
            pred = int(np.mean(nearest_labels) > 0.5)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def generate_predictions(self):
        """Generate and save predictions for all test embeddings."""
        print("Loading test embeddings...")
        test_path = os.path.join(self.test_dir, 'test_embeddings.pkl')
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        
        # Generate clusters if not already generated
        if not self.reference_data:
            self.generate_multilevel_clusters()
        
        print("Generating predictions...")
        all_predictions = []
        
        # Group test data by protein type
        unique_proteins = np.unique(test_data['protein_ids'])
        for protein_id in tqdm(unique_proteins):
            # Get embeddings for current protein
            mask = test_data['protein_ids'] == protein_id
            protein_embeddings = test_data['embeddings'][mask]
            molecule_ids = test_data['molecule_ids'][mask]
            
            # Generate predictions
            predictions = self.predict_binding(protein_embeddings, protein_id)
            
            # Store results
            for mol_id, pred in zip(molecule_ids, predictions):
                all_predictions.append({
                    'id': int(mol_id),
                    'binds': int(pred)
                })
        
        # Create and save submission file
        os.makedirs(self.output_dir, exist_ok=True)
        submission_df = pd.DataFrame(all_predictions)
        submission_df = submission_df.sort_values('id')
        submission_path = os.path.join(self.output_dir, 'submission.csv')
        submission_df.to_csv(submission_path, index=False)
        
        print(f"Predictions saved to {submission_path}")

if __name__ == "__main__":
    # Create prediction generator
    generator = PredictionGenerator(
        reference_dir='reference_embeddings',
        test_dir='test_embeddings',
        output_dir='submissions',
        k_neighbors=5
    )
    
    # Optimize parameters for each protein type
    generator.optimize_protein_parameters()
    
    # Generate predictions using optimized parameters
    generator.generate_predictions()
