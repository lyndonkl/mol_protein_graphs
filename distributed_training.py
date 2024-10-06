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

# Constants
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class ProteinProcessor:
    @staticmethod
    def residue_name_to_idx(res_name_one):
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G',
                       'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                       'T', 'W', 'Y', 'V']
        return amino_acids.index(res_name_one) if res_name_one in amino_acids else len(amino_acids)

    @staticmethod
    def process_protein(pdb_file: str, threshold: float = 5.0) -> HeteroData:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)

        amino_acids = [residue for model in structure for chain in model for residue in chain if is_aa(residue)]
        amino_acid_types = [index_to_one(three_to_index(residue.get_resname())) for residue in amino_acids]
        unique_amino_acids = list(set(amino_acid_types))

        data = HeteroData()

        node_features = {aa_type: [] for aa_type in unique_amino_acids}
        node_positions = {aa_type: [] for aa_type in unique_amino_acids}

        for residue, aa_type in zip(amino_acids, amino_acid_types):
            try:
                pos = residue['CA'].get_coord()
            except KeyError:
                pos = [0.0, 0.0, 0.0]
            node_features[aa_type].append([ProteinProcessor.residue_name_to_idx(aa_type)])
            node_positions[aa_type].append(pos)

        for aa_type in unique_amino_acids:
            data[aa_type].x = torch.tensor(node_features[aa_type], dtype=torch.float)
            data[aa_type].pos = torch.tensor(np.array(node_positions[aa_type]), dtype=torch.float)

        contact_edge_index = {}
        edge_types = set()
        reverse_edge_types = set()

        global_to_local_idx = {aa_type: {} for aa_type in unique_amino_acids}
        current_idx = {aa_type: 0 for aa_type in unique_amino_acids}

        for idx, aa_type in enumerate(amino_acid_types):
            global_to_local_idx[aa_type][idx] = current_idx[aa_type]
            current_idx[aa_type] += 1

        num_residues = len(amino_acids)
        for i in range(num_residues):
            residue_i, aa_i = amino_acids[i], amino_acid_types[i]
            try:
                pos_i = residue_i['CA'].get_coord()
            except KeyError:
                continue
            for j in range(i + 1, num_residues):
                residue_j, aa_j = amino_acids[j], amino_acid_types[j]
                try:
                    pos_j = residue_j['CA'].get_coord()
                except KeyError:
                    continue

                if np.linalg.norm(pos_i - pos_j) <= threshold:
                    edge_type = (aa_i, 'contact', aa_j) if aa_i <= aa_j else (aa_j, 'contact', aa_i)
                    src_aa, tgt_aa = edge_type[0], edge_type[2]
                    src_global, tgt_global = (i, j) if aa_i <= aa_j else (j, i)

                    if edge_type not in contact_edge_index:
                        contact_edge_index[edge_type] = []

                    src_local = global_to_local_idx[src_aa][src_global]
                    tgt_local = global_to_local_idx[tgt_aa][tgt_global]

                    contact_edge_index[edge_type].append([src_local, tgt_local])
                    edge_types.add(edge_type)

        for edge_type, edges in contact_edge_index.items():
            if edges:
                src_type, relation, tgt_type = edge_type
                reverse_edge_type = (tgt_type, relation, src_type)
                reverse_edge_types.add(reverse_edge_type)

                edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[edge_type].edge_index = edge_tensor

                reverse_edges = [[tgt, src] for src, tgt in edges]
                reverse_edge_tensor = torch.tensor(reverse_edges, dtype=torch.long).t().contiguous()
                data[reverse_edge_type].edge_index = reverse_edge_tensor

        data.node_types = set(unique_amino_acids)
        data.edge_types = edge_types
        data.reverse_edge_types = reverse_edge_types

        return data

class CrossAttentionLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super(CrossAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.W_Q = Linear(hidden_dim, hidden_dim)
        self.W_K = Linear(hidden_dim, hidden_dim)
        self.W_V = Linear(hidden_dim, hidden_dim)

        self.scale = self.head_dim ** 0.5

    def forward(self, query_nodes: torch.Tensor, key_nodes: torch.Tensor) -> torch.Tensor:
        N_q, N_k = query_nodes.size(0), key_nodes.size(0)

        Q = self.W_Q(query_nodes).view(N_q, self.num_heads, self.head_dim).permute(1, 0, 2)
        K = self.W_K(key_nodes).view(N_k, self.num_heads, self.head_dim).permute(1, 0, 2)
        V = self.W_V(key_nodes).view(N_k, self.num_heads, self.head_dim).permute(1, 0, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, V).permute(1, 0, 2).contiguous().view(N_q, -1)

        return out

class CrossGraphAttentionModel(torch.nn.Module):
    def __init__(self, hidden_dim: int = 64, num_attention_heads: int = 4):
        super(CrossGraphAttentionModel, self).__init__()

        self.mol_conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in molecule_edge_types
        }, aggr='mean')

        self.mol_conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in molecule_edge_types
        }, aggr='mean')

        self.prot_conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in protein_edge_types
        }, aggr='mean')

        self.prot_conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in protein_edge_types
        }, aggr='mean')

        self.cross_attn_mol_to_prot = CrossAttentionLayer(hidden_dim, num_attention_heads)
        self.cross_attn_prot_to_mol = CrossAttentionLayer(hidden_dim, num_attention_heads)

        self.fc1 = Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = Linear(hidden_dim, 1)

    def forward(self, mol_data: HeteroData, prot_data: HeteroData) -> torch.Tensor:
        x_mol_dict = mol_data.x_dict
        edge_index_mol_dict = mol_data.edge_index_dict

        x_mol_dict = {key: F.relu(x) for key, x in self.mol_conv1(x_mol_dict, edge_index_mol_dict).items()}
        x_mol_dict = {key: F.relu(x) for key, x in self.mol_conv2(x_mol_dict, edge_index_mol_dict).items()}

        H_mol = torch.cat([x_mol_dict[nt] for nt in molecule_node_types if nt in x_mol_dict], dim=0)

        x_prot_dict = prot_data.x_dict
        edge_index_prot_dict = prot_data.edge_index_dict

        x_prot_dict = {key: F.relu(x) for key, x in self.prot_conv1(x_prot_dict, edge_index_prot_dict).items()}
        x_prot_dict = {key: F.relu(x) for key, x in self.prot_conv2(x_prot_dict, edge_index_prot_dict).items()}

        H_prot = torch.cat([x_prot_dict[nt] for nt in protein_node_types if nt in x_prot_dict], dim=0)

        H_mol_attn = self.cross_attn_mol_to_prot(H_mol, H_prot)
        H_prot_attn = self.cross_attn_prot_to_mol(H_prot, H_mol)

        H_mol_combined = H_mol + H_mol_attn
        H_prot_combined = H_prot + H_prot_attn

        mol_batches = torch.cat([mol_data.batch_dict[nt] for nt in molecule_node_types if nt in mol_data.batch_dict])
        prot_batches = torch.cat([prot_data.batch_dict[nt] for nt in protein_node_types if nt in prot_data.batch_dict])

        z_mol = global_mean_pool(H_mol_combined, mol_batches)
        z_prot = global_mean_pool(H_prot_combined, prot_batches)

        z_joint = torch.cat([z_mol, z_prot], dim=1)

        x = F.relu(self.fc1(z_joint))
        out = torch.sigmoid(self.fc2(x))

        return out.squeeze()

def custom_transform(batch):
    mol_batch, prot_batch, batch_size = collate_fn(batch)
    return {
        'mol_batch': mol_batch,
        'prot_batch': prot_batch,
        'batch_size': batch_size
    }

def collate_fn(batch):
    valid_items = [item for item in batch if item is not None and item[0] is not None and item[0]['invalid'] is False]
    
    mol_batch = [item[0] for item in valid_items]
    prot_batch = [item[1] for item in valid_items]

    mol_batch = Batch.from_data_list(mol_batch)
    prot_batch = Batch.from_data_list(prot_batch)

    batch_size = len(valid_items)

    return mol_batch, prot_batch, batch_size

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

def setup_logger():
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


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
    trainer = Trainer(model, train_loader, val_loader, test_loader, criterion, optimizer, rank, world_size)

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
