import warnings
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
import torch.nn.functional as F
from torch.utils.data import Subset
import pandas as pd
import numpy as np
import torch
import os
from datasets import CombinedDataset, MoleculeDataset
import json
import warnings
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import csv

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Dataset, HeteroData, DataLoader
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import three_to_index, index_to_one
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import numpy as np
import random
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# Set random seed and device
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, GCNConv, Linear, global_mean_pool
from torch.utils.data import random_split
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import three_to_index, index_to_one
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper functions
def residue_name_to_idx(res_name_one):
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G',
                   'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                   'T', 'W', 'Y', 'V']
    if res_name_one in amino_acids:
        return amino_acids.index(res_name_one)
    else:
        return len(amino_acids)  # Unknown amino acid

def process_protein(pdb_file, threshold=5.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    amino_acids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue):
                    amino_acids.append(residue)

    amino_acid_types = [index_to_one(three_to_index(residue.get_resname())) for residue in amino_acids]
    unique_amino_acids = list(set(amino_acid_types))

    data = HeteroData()

    node_features = {}
    node_positions = {}
    node_counter = 0

    # Initialize node features and positions
    for aa_type in unique_amino_acids:
        node_features[aa_type] = []
        node_positions[aa_type] = []

    for idx, (residue, aa_type) in enumerate(zip(amino_acids, amino_acid_types)):
        try:
            ca_atom = residue['CA']
            pos = ca_atom.get_coord()
        except KeyError:
            pos = [0.0, 0.0, 0.0]
        node_features[aa_type].append([residue_name_to_idx(aa_type)])
        node_positions[aa_type].append(pos)
        node_counter += 1

    for aa_type in unique_amino_acids:
        data[aa_type].x = torch.tensor(node_features[aa_type], dtype=torch.float)
        data[aa_type].pos = torch.tensor(np.array(node_positions[aa_type]), dtype=torch.float)

    # Build edges based on proximity
    contact_edge_index = {}
    edge_types = set()
    reverse_edge_types = set()

    # Mapping from global index to local index within node type
    global_to_local_idx = {}
    current_idx = {aa_type: 0 for aa_type in unique_amino_acids}

    for aa_type in amino_acid_types:
        global_to_local_idx[aa_type] = {}

    for idx, aa_type in enumerate(amino_acid_types):
        global_idx = idx
        local_idx = current_idx[aa_type]
        global_to_local_idx[aa_type][global_idx] = local_idx
        current_idx[aa_type] += 1

    num_residues = len(amino_acids)
    for i in range(num_residues):
        residue_i = amino_acids[i]
        aa_i = amino_acid_types[i]
        try:
            ca_i = residue_i['CA']
            pos_i = ca_i.get_coord()
        except KeyError:
            continue
        for j in range(i + 1, num_residues):  # Ensure j > i to avoid duplicates
            residue_j = amino_acids[j]
            aa_j = amino_acid_types[j]
            try:
                ca_j = residue_j['CA']
                pos_j = ca_j.get_coord()
            except KeyError:
                continue

            distance = np.linalg.norm(pos_i - pos_j)
            if distance <= threshold:
                # Define edge type in consistent order
                if aa_i <= aa_j:
                    edge_type = (aa_i, 'contact', aa_j)
                    src_aa, tgt_aa = aa_i, aa_j
                    src_global, tgt_global = i, j
                else:
                    edge_type = (aa_j, 'contact', aa_i)
                    src_aa, tgt_aa = aa_j, aa_i
                    src_global, tgt_global = j, i

                # Initialize edge list if not present
                if edge_type not in contact_edge_index:
                    contact_edge_index[edge_type] = []

                # Get local indices within their respective node types
                src_local = global_to_local_idx[src_aa][src_global]
                tgt_local = global_to_local_idx[tgt_aa][tgt_global]

                # Append edge
                contact_edge_index[edge_type].append([src_local, tgt_local])
                edge_types.add(edge_type)

    # Assign edges to HeteroData
    for edge_type, edges in contact_edge_index.items():
        if len(edges) > 0:
            # Extract the original source and target types
            src_type, relation, tgt_type = edge_type
    
            # Create reverse edge type
            reverse_edge_type = (tgt_type, relation, src_type)
            reverse_edge_types.add(reverse_edge_type)
    
            # Convert original edges to tensor
            edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
            # Assign original edges to original edge type
            data[edge_type].edge_index = edge_tensor
    
            # Create reverse edges
            reverse_edges = [[tgt, src] for src, tgt in edges]
            reverse_edge_tensor = torch.tensor(reverse_edges, dtype=torch.long).t().contiguous()
    
            # Assign reverse edges to reverse edge type
            data[reverse_edge_type].edge_index = reverse_edge_tensor

    data.node_types = set(unique_amino_acids)
    data.edge_types = edge_types
    data.reverse_edge_types = reverse_edge_types

    return data

from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd

def collect_protein_node_and_edge_types(protein_graphs):
    protein_node_types = set()
    protein_edge_types = set()
    for protein_data in protein_graphs.values():
        protein_node_types.update(protein_data.node_types)
        protein_edge_types.update(protein_data.edge_types)
        protein_edge_types.update(protein_data.reverse_edge_types)
    return sorted(protein_node_types), sorted(protein_edge_types)

# Process and store protein graphs
protein_graphs = {}
protein_pdb_files = {
    'BRD4': './BRD4.pdb',
    'HSA': './ALB.pdb',
    'sEH': './EPH.pdb'
}

for protein_name, pdb_file in protein_pdb_files.items():
    if os.path.exists(pdb_file):
        protein_data = process_protein(pdb_file)
        protein_graphs[protein_name] = protein_data
    else:
        print(f"PDB file {pdb_file} for {protein_name} does not exist.")

# Load the unique atom and edge types from the JSON file
with open('unique_atom_and_edge_types.json', 'r') as f:
    unique_types = json.load(f)

# Extract molecule node and edge types
molecule_node_types = unique_types['molecule_node_types']
molecule_edge_types = [tuple(edge) for edge in unique_types['molecule_edge_types']]

# Now molecule_node_types and molecule_edge_types can be used in your code
print("Collected molecule node and edge types successfully.")

print("Collecting protein node and edge types...")
protein_node_types, protein_edge_types = collect_protein_node_and_edge_types(protein_graphs)

# Load your external test dataset
external_test_df = pd.read_parquet('test.parquet')
external_test_dataset = CombinedDataset(external_test_df, protein_graphs)

# Custom collate function
def collate_fn(batch):
    print('Here')
    print(item[0]['invalid'])
    mol_batch = [item[0] for item in batch if item is not None and item[0] is not None and getattr(item[0], 'invalid', False) is False]
    prot_batch = [item[1] for item in batch if item is not None and item[0] is not None and getattr(item[0], 'invalid', False) is False]

    mol_batch = Batch.from_data_list(mol_batch)
    prot_batch = Batch.from_data_list(prot_batch)

    return mol_batch, prot_batch

# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

class CrossAttentionLayer(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(CrossAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.W_Q = torch.nn.Linear(hidden_dim, hidden_dim)
        self.W_K = torch.nn.Linear(hidden_dim, hidden_dim)
        self.W_V = torch.nn.Linear(hidden_dim, hidden_dim)

        self.scale = self.head_dim ** 0.5

    def forward(self, query_nodes, key_nodes):
        # query_nodes: [N_q, hidden_dim]
        # key_nodes: [N_k, hidden_dim]

        N_q = query_nodes.size(0)
        N_k = key_nodes.size(0)

        Q = self.W_Q(query_nodes)  # [N_q, hidden_dim]
        K = self.W_K(key_nodes)    # [N_k, hidden_dim]
        V = self.W_V(key_nodes)    # [N_k, hidden_dim]

        # Reshape for multi-head attention
        Q = Q.view(N_q, self.num_heads, self.head_dim).permute(1, 0, 2)  # [num_heads, N_q, head_dim]
        K = K.view(N_k, self.num_heads, self.head_dim).permute(1, 0, 2)  # [num_heads, N_k, head_dim]
        V = V.view(N_k, self.num_heads, self.head_dim).permute(1, 0, 2)  # [num_heads, N_k, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [num_heads, N_q, N_k]
        attn_weights = torch.softmax(attn_scores, dim=-1)                # [num_heads, N_q, N_k]

        # Compute attended values
        out = torch.matmul(attn_weights, V)  # [num_heads, N_q, head_dim]
        out = out.permute(1, 0, 2).contiguous().view(N_q, -1)  # [N_q, hidden_dim]

        return out

class CrossGraphAttentionModel(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_attention_heads=4):
        super(CrossGraphAttentionModel, self).__init__()

        # print(molecule_edge_types)

        # Molecule GNN Encoder
        self.mol_conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in molecule_edge_types
        }, aggr='mean')

        self.mol_conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in molecule_edge_types
        }, aggr='mean')

        # Protein GNN Encoder
        self.prot_conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in protein_edge_types
        }, aggr='mean')

        self.prot_conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in protein_edge_types
        }, aggr='mean')

        # Cross-Attention Layers
        self.cross_attn_mol_to_prot = CrossAttentionLayer(hidden_dim, num_attention_heads)
        self.cross_attn_prot_to_mol = CrossAttentionLayer(hidden_dim, num_attention_heads)

        # Fully Connected Layers
        self.fc1 = Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = Linear(hidden_dim, 1)

    def forward(self, mol_data, prot_data):
        # Molecule GNN Encoding
        x_mol_dict = mol_data.x_dict
        edge_index_mol_dict = mol_data.edge_index_dict

        x_mol_dict = self.mol_conv1(x_mol_dict, edge_index_mol_dict)
        x_mol_dict = {key: F.relu(x) for key, x in x_mol_dict.items()}

        x_mol_dict = self.mol_conv2(x_mol_dict, edge_index_mol_dict)
        x_mol_dict = {key: F.relu(x) for key, x in x_mol_dict.items()}

        # Concatenate molecule node embeddings
        mol_node_embeddings = []
        for nt in molecule_node_types:
            if nt in x_mol_dict:
                mol_node_embeddings.append(x_mol_dict[nt])
        H_mol = torch.cat(mol_node_embeddings, dim=0)

        # Protein GNN Encoding
        x_prot_dict = prot_data.x_dict
        edge_index_prot_dict = prot_data.edge_index_dict

        x_prot_dict = self.prot_conv1(x_prot_dict, edge_index_prot_dict)
        x_prot_dict = {key: F.relu(x) for key, x in x_prot_dict.items()}

        x_prot_dict = self.prot_conv2(x_prot_dict, edge_index_prot_dict)
        x_prot_dict = {key: F.relu(x) for key, x in x_prot_dict.items()}

        # Concatenate protein node embeddings
        prot_node_embeddings = []
        for nt in protein_node_types:
            if nt in x_prot_dict:
                prot_node_embeddings.append(x_prot_dict[nt])
        H_prot = torch.cat(prot_node_embeddings, dim=0)

        # Cross-Attention
        H_mol_attn = self.cross_attn_mol_to_prot(H_mol, H_prot)
        H_prot_attn = self.cross_attn_prot_to_mol(H_prot, H_mol)

        # Combine original and attended embeddings
        H_mol_combined = H_mol + H_mol_attn
        H_prot_combined = H_prot + H_prot_attn

        # # Global Pooling
        # mol_batch = mol_data.batch if hasattr(mol_data, 'batch') else torch.zeros(H_mol_combined.size(0), dtype=torch.long, device=H_mol_combined.device)
        # prot_batch = prot_data.batch if hasattr(prot_data, 'batch') else torch.zeros(H_prot_combined.size(0), dtype=torch.long, device=H_prot_combined.device)

        # Global Pooling
        # Use batch_dict to get batch information per node type
        mol_batches = torch.cat([mol_data.batch_dict[nt] for nt in molecule_node_types if nt in mol_data.batch_dict])
        prot_batches = torch.cat([prot_data.batch_dict[nt] for nt in protein_node_types if nt in prot_data.batch_dict])

        # z_mol = global_mean_pool(H_mol_combined, mol_batch)
        # z_prot = global_mean_pool(H_prot_combined, prot_batch)

        z_mol = global_mean_pool(H_mol_combined, mol_batches)
        z_prot = global_mean_pool(H_prot_combined, prot_batches)

        # Joint Representation
        z_joint = torch.cat([z_mol, z_prot], dim=1)

        # Prediction
        x = F.relu(self.fc1(z_joint))
        out = torch.sigmoid(self.fc2(x))

        return out.squeeze()

def process_chunk(idx, model):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    try:
        data = external_test_dataset[idx]
        if getattr(data[0], 'invalid', False):
            print(f"Skipping processing index {idx}")
            return None

        mol_data = data[0].to(device)
        prot_data= data[1].to(device)
        out = model(mol_data, prot_data)
        ids = mol_data['smolecule'].id.cpu().numpy()
        predictions = out.cpu().numpy()
        results = [{'id': int(id_val), 'binds': float(pred)} for id_val, pred in zip(ids, predictions)]
        return results
    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        return None

def external_predict_parallel(model, dataset, output_csv_path, n_jobs=-1):
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count() + 1 + n_jobs
    n_jobs = max(n_jobs, 1)

    total_samples = len(dataset)
    indices = list(range(total_samples))

    with open(output_csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['id', 'binds']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(process_chunk, idx, model): idx for idx in indices}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                result = future.result()
                if result:
                    writer.writerows(result)

    print(f"Predictions have been written to {output_csv_path}")

# Usage
if __name__ == '__main__':
    # Define the model
    # Suppress all FutureWarning messages
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # Specify the device
    device = torch.device('cpu')  # Use 'cuda' if you have a GPU available
    
    # Load the state dictionary
    state_dict = torch.load('cross_graph_attention_model.pth', map_location=device, weights_only=True)
    model = CrossGraphAttentionModel(hidden_dim=64, num_attention_heads=4)
    state_dict = torch.load('cross_graph_attention_model.pth', map_location=device)
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    external_test_dataset = CombinedDataset(external_test_df, protein_graphs)
    external_predict_parallel(model, external_test_dataset, output_csv_path='submissions.csv', n_jobs=-1)