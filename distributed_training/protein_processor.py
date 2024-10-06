# Standard library imports

# Third-party imports
import numpy as np
import torch
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import three_to_index, index_to_one
from torch_geometric.data import HeteroData

class ProteinProcessor:
    @staticmethod
    def residue_name_to_idx(res_name_one):
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G',
                       'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                       'T', 'W', 'Y', 'V']
        return amino_acids.index(res_name_one) if res_name_one in amino_acids else len(amino_acids)

    @staticmethod
    def process_protein(pdb_file: str, threshold: float = 5.0) -> HeteroData:
        # Initialize PDB parser and load structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)

        # Extract amino acids and their types
        amino_acids = [residue for model in structure for chain in model for residue in chain if is_aa(residue)]
        amino_acid_types = [index_to_one(three_to_index(residue.get_resname())) for residue in amino_acids]
        unique_amino_acids = list(set(amino_acid_types))

        # Initialize HeteroData object
        data = HeteroData()

        # Prepare node features and positions
        node_features = {aa_type: [] for aa_type in unique_amino_acids}
        node_positions = {aa_type: [] for aa_type in unique_amino_acids}

        for residue, aa_type in zip(amino_acids, amino_acid_types):
            try:
                pos = residue['CA'].get_coord()
            except KeyError:
                pos = [0.0, 0.0, 0.0]
            node_features[aa_type].append([ProteinProcessor.residue_name_to_idx(aa_type)])
            node_positions[aa_type].append(pos)

        # Add node features and positions to HeteroData object
        for aa_type in unique_amino_acids:
            data[aa_type].x = torch.tensor(node_features[aa_type], dtype=torch.float)
            data[aa_type].pos = torch.tensor(np.array(node_positions[aa_type]), dtype=torch.float)

        # Initialize edge data structures
        contact_edge_index = {}
        edge_types = set()
        reverse_edge_types = set()

        # Create mapping from global to local indices
        global_to_local_idx = {aa_type: {} for aa_type in unique_amino_acids}
        current_idx = {aa_type: 0 for aa_type in unique_amino_acids}

        for idx, aa_type in enumerate(amino_acid_types):
            global_to_local_idx[aa_type][idx] = current_idx[aa_type]
            current_idx[aa_type] += 1

        # Compute edges based on distance threshold
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

        # Add edges and their reverse to HeteroData object
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

        # Add metadata to HeteroData object
        data.node_types = set(unique_amino_acids)
        data.edge_types = edge_types
        data.reverse_edge_types = reverse_edge_types

        return data