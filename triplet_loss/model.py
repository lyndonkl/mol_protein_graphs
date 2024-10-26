# Standard library imports

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch.nn import Linear

# Custom imports

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super(SelfAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.W_Q = Linear(hidden_dim, hidden_dim)
        self.W_K = Linear(hidden_dim, hidden_dim)
        self.W_V = Linear(hidden_dim, hidden_dim)

        self.scale = self.head_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)

        # Reshape and permute for multi-head attention
        Q = self.W_Q(x).view(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        K = self.W_K(x).view(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        V = self.W_V(x).view(N, self.num_heads, self.head_dim).permute(1, 0, 2)

        # Compute attention scores and weights
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply attention weights to values and reshape
        out = torch.matmul(attn_weights, V).permute(1, 0, 2).contiguous().view(N, -1)

        return out

class MoleculeGraphTripletBlock(nn.Module):
    def __init__(self, graph_metadata, hidden_dim: int = 64, num_attention_heads: int = 4):
        super(MoleculeGraphTripletBlock, self).__init__()

        self.molecule_node_types = graph_metadata['molecule_node_types']
        self.molecule_edge_types = graph_metadata['molecule_edge_types']

        # Define molecule graph convolutions
        self.mol_conv = HeteroConv({
            edge_type: GINEConv(
                nn=nn.Sequential(
                    Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    Linear(hidden_dim, hidden_dim)
                )
            )
            for edge_type in self.molecule_edge_types
        }, aggr='mean')

        # Define self-attention layer
        self.self_attention = SelfAttentionLayer(hidden_dim, num_attention_heads)

    def forward(self, mol_data: HeteroData, protein_emb: torch.Tensor) -> torch.Tensor:
        # Prepare edge attribute dictionary
        edge_attr_mol_dict = {edge_type: mol_data[edge_type].edge_attr for edge_type in mol_data.edge_types}

        # Process molecule graph with GNN layer
        x_mol_dict = mol_data.x_dict
        edge_index_mol_dict = mol_data.edge_index_dict

        x_mol_dict = self.mol_conv(x_mol_dict, edge_index_mol_dict, edge_attr_dict=edge_attr_mol_dict)
        x_mol_dict = {key: F.relu(x) for key, x in x_mol_dict.items()}

        H_mol = torch.cat([x_mol_dict[nt] for nt in self.molecule_node_types if nt in x_mol_dict], dim=0)

        # Apply self-attention
        H_mol = self.self_attention(H_mol)

        # Global pooling for molecule graph
        mol_batches = torch.cat([mol_data.batch_dict[nt] for nt in self.molecule_node_types if nt in mol_data.batch_dict])
        z_mol = global_mean_pool(H_mol, mol_batches)

        # Combine molecule graph embedding with protein embedding
        z_combined = z_mol + protein_emb

        return z_combined

class StackedMoleculeGraphTripletModel(nn.Module):
    def __init__(self, num_proteins: int, graph_metadata, hidden_dim: int = 64, num_attention_heads: int = 4, num_layers: int = 3):
        super(StackedMoleculeGraphTripletModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        mol_node_input_dim = 11  # Molecule node features (e.g., 8 features + 3 positions)
        mol_edge_input_dim = 10  # Molecule edge attributes

        # Protein embedding
        self.protein_embedding = nn.Embedding(num_proteins, hidden_dim)

        # Linear layers for initial node features
        self.initial_node_lin = nn.ModuleDict()
        for node_type in graph_metadata['molecule_node_types']:
            self.initial_node_lin[node_type] = Linear(mol_node_input_dim, hidden_dim)

        # Linear layers for initial edge attributes
        self.initial_edge_lin = nn.ModuleDict()
        for edge_type in graph_metadata['molecule_edge_types']:
            edge_type_str = '__'.join(edge_type)
            self.initial_edge_lin[edge_type_str] = Linear(mol_edge_input_dim, hidden_dim)

        # Create multiple MoleculeGraphTripletBlocks
        self.blocks = nn.ModuleList([
            MoleculeGraphTripletBlock(graph_metadata, hidden_dim, num_attention_heads) for _ in range(num_layers)
        ])

        # Final linear layers for triplet embedding
        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)

    def forward(self, mol_data: HeteroData, protein_type: torch.Tensor) -> torch.Tensor:
        # Initial node feature mapping
        for node_type in mol_data.node_types:
            if node_type != 'smolecule':
                x = mol_data[node_type].x
                mol_data[node_type].x = self.initial_node_lin[node_type](x)

        # Initial edge attribute mapping
        for edge_type in mol_data.edge_types:
            edge_type_str = '__'.join(edge_type)
            if 'edge_attr' in mol_data[edge_type]:
                edge_attr = mol_data[edge_type].edge_attr
                mol_data[edge_type].edge_attr = self.initial_edge_lin[edge_type_str](edge_attr)

        # Apply protein embedding
        protein_emb = self.protein_embedding(protein_type)

        # Sequentially pass data through each block
        z_combined = protein_emb
        for block in self.blocks:
            z_combined = block(mol_data, z_combined)

        # Final embedding for triplet loss
        x = F.relu(self.fc1(z_combined))
        out_embedding = self.fc2(x)

        return out_embedding
