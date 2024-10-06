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