# Standard library imports

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GINEConv, global_mean_pool
from torch.nn import Linear

# Custom imports
from .layer import SelfAttentionLayer

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
