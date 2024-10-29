import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import HeteroData

from triplet_loss.src.model.block import MoleculeGraphTripletBlock

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