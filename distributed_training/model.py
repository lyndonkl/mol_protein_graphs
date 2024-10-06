# Standard library imports

# Third-party imports
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool

# Custom imports

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

        # Reshape and permute for multi-head attention
        Q = self.W_Q(query_nodes).view(N_q, self.num_heads, self.head_dim).permute(1, 0, 2)
        K = self.W_K(key_nodes).view(N_k, self.num_heads, self.head_dim).permute(1, 0, 2)
        V = self.W_V(key_nodes).view(N_k, self.num_heads, self.head_dim).permute(1, 0, 2)

        # Compute attention scores and weights
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply attention weights to values and reshape
        out = torch.matmul(attn_weights, V).permute(1, 0, 2).contiguous().view(N_q, -1)

        return out

class CrossGraphAttentionModel(torch.nn.Module):
    def __init__(self, graph_metadata, hidden_dim: int = 64, num_attention_heads: int = 4):
        super(CrossGraphAttentionModel, self).__init__()

        self.molecule_node_types = graph_metadata['molecule_node_types']
        self.protein_node_types = graph_metadata['protein_node_types']
        self.molecule_edge_types = graph_metadata['molecule_edge_types']
        self.protein_edge_types = graph_metadata['protein_edge_types']

        # Define molecule graph convolutions
        self.mol_conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in self.molecule_edge_types
        }, aggr='mean')

        self.mol_conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in self.molecule_edge_types
        }, aggr='mean')

        # Define protein graph convolutions
        self.prot_conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in self.protein_edge_types
        }, aggr='mean')

        self.prot_conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in self.protein_edge_types
        }, aggr='mean')

        # Define cross-attention layers
        self.cross_attn_mol_to_prot = CrossAttentionLayer(hidden_dim, num_attention_heads)
        self.cross_attn_prot_to_mol = CrossAttentionLayer(hidden_dim, num_attention_heads)

        # Define fully connected layers
        self.fc1 = Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = Linear(hidden_dim, 1)

    def forward(self, mol_data: HeteroData, prot_data: HeteroData) -> torch.Tensor:
        # Process molecule graph
        x_mol_dict = mol_data.x_dict
        edge_index_mol_dict = mol_data.edge_index_dict

        x_mol_dict = {key: F.relu(x) for key, x in self.mol_conv1(x_mol_dict, edge_index_mol_dict).items()}
        x_mol_dict = {key: F.relu(x) for key, x in self.mol_conv2(x_mol_dict, edge_index_mol_dict).items()}

        H_mol = torch.cat([x_mol_dict[nt] for nt in self.molecule_node_types if nt in x_mol_dict], dim=0)

        # Process protein graph
        x_prot_dict = prot_data.x_dict
        edge_index_prot_dict = prot_data.edge_index_dict

        x_prot_dict = {key: F.relu(x) for key, x in self.prot_conv1(x_prot_dict, edge_index_prot_dict).items()}
        x_prot_dict = {key: F.relu(x) for key, x in self.prot_conv2(x_prot_dict, edge_index_prot_dict).items()}

        H_prot = torch.cat([x_prot_dict[nt] for nt in self.protein_node_types if nt in x_prot_dict], dim=0)

        # Apply cross-attention
        H_mol_attn = self.cross_attn_mol_to_prot(H_mol, H_prot)
        H_prot_attn = self.cross_attn_prot_to_mol(H_prot, H_mol)

        H_mol_combined = H_mol + H_mol_attn
        H_prot_combined = H_prot + H_prot_attn

        # Global pooling
        mol_batches = torch.cat([mol_data.batch_dict[nt] for nt in self.molecule_node_types if nt in mol_data.batch_dict])
        prot_batches = torch.cat([prot_data.batch_dict[nt] for nt in self.protein_node_types if nt in prot_data.batch_dict])

        z_mol = global_mean_pool(H_mol_combined, mol_batches)
        z_prot = global_mean_pool(H_prot_combined, prot_batches)

        # Final prediction
        z_joint = torch.cat([z_mol, z_prot], dim=1)
        x = F.relu(self.fc1(z_joint))
        out = torch.sigmoid(self.fc2(x))

        return out.squeeze()