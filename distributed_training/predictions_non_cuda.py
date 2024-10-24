import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from tqdm import tqdm
import csv
from datasets import CombinedDataset
from protein_processor import ProteinProcessor
from model import StackedCrossGraphAttentionModel
from utils import setup_logger, collect_protein_node_and_edge_types, collate_fn

# Constants
RANDOM_SEED = 42
DEVICE = torch.device('cpu')

# Set environment variables
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.multiprocessing.set_sharing_strategy('file_system')

class Predictor:
    def __init__(self, model, test_dataset, rank, world_size):
        self.logger = setup_logger()
        self.logger.info(f"[Rank {rank}] Initializing Predictor")

        self.rank = rank
        self.world_size = world_size
        self.model = model.to(DEVICE)

        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            num_workers=2,
            shuffle=False,
            sampler=test_sampler,
            collate_fn=collate_fn
        )

    def predict(self, output_csv_path):
        self.logger.info(f"[Rank {self.rank}] Starting predictions")
        self.model.eval()
        
        predictions = []
        ids = []

        with torch.no_grad():
            for mol_data, prot_data, batch_size in tqdm(self.test_loader, desc=f"Predicting (Rank {self.rank})", disable=(self.rank != 0)):
                mol_data = mol_data.to(DEVICE)
                prot_data = prot_data.to(DEVICE)
                out = self.model(mol_data, prot_data)

                batch_ids = mol_data['smolecule'].id.cpu().numpy()
                batch_predictions = out.cpu().numpy()

                ids.extend(batch_ids)
                predictions.extend(batch_predictions)

        self.logger.info(f"[Rank {self.rank}] Finished predictions")

        # Gather predictions and ids from all processes
        all_predictions = [torch.zeros_like(torch.tensor(predictions), device=DEVICE) for _ in range(self.world_size)]
        all_ids = [torch.zeros_like(torch.tensor(ids), device=DEVICE) for _ in range(self.world_size)]

        dist.all_gather(all_predictions, torch.tensor(predictions))
        dist.all_gather(all_ids, torch.tensor(ids))

        if self.rank == 0:
            # Flatten the gathered lists
            all_predictions = [item for sublist in all_predictions for item in sublist]
            all_ids = [item for sublist in all_ids for item in sublist]

            # Write results to CSV
            with open(output_csv_path, mode='w', newline='') as csv_file:
                fieldnames = ['id', 'binds']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                
                for id_val, pred in zip(all_ids, all_predictions):
                    writer.writerow({'id': int(id_val), 'binds': float(pred)})

            self.logger.info(f"Predictions have been written to {output_csv_path}")

def run_predictions(rank: int, world_size: int, test_dataset, graph_metadata, model_path, output_csv_path):
    logger = setup_logger()
    logger.info(f"[Rank {rank}] Starting run_predictions function")
    
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    logger.info(f"[Rank {rank}] Process group initialized with backend gloo")

    # Initialize model
    model = StackedCrossGraphAttentionModel(graph_metadata, hidden_dim=128, num_attention_heads=8, num_layers=4)
    logger.info(f"[Rank {rank}] Model initialized")

    # Load model weights
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    logger.info(f"[Rank {rank}] Model weights loaded from {model_path}")

    # Run predictions
    predictor = Predictor(model, test_dataset, rank, world_size)
    predictor.predict(output_csv_path)

    dist.destroy_process_group()
    logger.info(f"[Rank {rank}] Destroyed process group and exiting")

def main():
    # Load test data
    test_df = pd.read_parquet('../cleaned_test.parquet')
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

    # Create dataset
    test_dataset = CombinedDataset(test_df, protein_graphs)

    # Set paths
    model_path = 'best_cross_graph_attention_model.pth'
    output_csv_path = 'submissions.csv'

    # For CPU, use the number of available cores. For MPS, use 1.
    world_size = mp.cpu_count() if DEVICE.type == 'cpu' else 1

    mp.spawn(run_predictions, args=(world_size, test_dataset, graph_metadata, model_path, output_csv_path), nprocs=world_size, join=True)

    # Fill in missing predictions
    df_csv = pd.read_csv(output_csv_path)
    csv_ids = set(df_csv['id'])
    parquet_ids = set(test_df['id'])
    missing_ids = parquet_ids - csv_ids

    missing_data = pd.DataFrame({'id': list(missing_ids), 'binds': [0.5] * len(missing_ids)})
    df_csv = pd.concat([df_csv, missing_data], ignore_index=True)
    df_csv.to_csv(output_csv_path, index=False)

    total_count = len(csv_ids) + len(missing_ids)
    print(f'Missing count: {len(missing_ids)}')
    print(f'Total count: {total_count}')

if __name__ == '__main__':
    main()
