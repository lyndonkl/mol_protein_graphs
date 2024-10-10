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
    def get_residue_features(residue):
        # Hydrophobicity scale from Kyte & Doolittle
        hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'E': -3.5, 'Q': -3.5, 'G': -0.4,
            'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
            'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }

        # Molecular weight of amino acids (in Da)
        molecular_weight = {
            'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2, 'E': 147.1, 'Q': 146.2, 'G': 75.1,
            'H': 155.2, 'I': 131.2, 'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1, 'S': 105.1,
            'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
        }

        aa_type = index_to_one(three_to_index(residue.get_resname()))
        
        # Extract features
        hydrophobicity = hydrophobicity_scale.get(aa_type, 0.0)
        charge = 1 if aa_type in ['R', 'K', 'H'] else -1 if aa_type in ['D', 'E'] else 0
        polarity = 1 if aa_type in ['N', 'Q', 'S', 'T', 'Y', 'C'] else 0
        weight = molecular_weight.get(aa_type, 0.0)
        is_aromatic = 1 if aa_type in ['F', 'Y', 'W'] else 0

        # New features
        is_polar = 1 if aa_type in ['R', 'N', 'D', 'E', 'Q', 'H', 'K', 'S', 'T', 'Y'] else 0
        is_aliphatic = 1 if aa_type in ['A', 'I', 'L', 'V'] else 0
        is_small = 1 if aa_type in ['A', 'G', 'S', 'C', 'T', 'P', 'D', 'N', 'V'] else 0
        is_tiny = 1 if aa_type in ['A', 'G', 'S', 'C'] else 0
        is_proline = 1 if aa_type == 'P' else 0
        is_cysteine = 1 if aa_type == 'C' else 0
        side_chain_length = len(residue.get_resname()) - 3  # Approximate side chain length

        return [
            hydrophobicity,
            charge,
            polarity,
            weight,
            is_aromatic,
            is_polar,
            is_aliphatic,
            is_small,
            is_tiny,
            is_proline,
            is_cysteine,
            side_chain_length
        ]

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
            node_features[aa_type].append(ProteinProcessor.get_residue_features(residue))
            node_positions[aa_type].append(pos)

        # Add node features and positions to HeteroData object
        for aa_type in unique_amino_acids:
            x_feats = torch.tensor(node_features[aa_type], dtype=torch.float)
            pos = torch.tensor(node_positions[aa_type], dtype=torch.float)
            # Concatenate node features with positions
            x = torch.cat([x_feats, pos], dim=1)
            data[f"{aa_type}_protein"].x = x

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
                res_id_i = residue_i.get_id()[1] 
            except KeyError:
                continue
            for j in range(i + 1, num_residues):
                residue_j, aa_j = amino_acids[j], amino_acid_types[j]
                try:
                    pos_j = residue_j['CA'].get_coord()
                    res_id_j = residue_j.get_id()[1]
                except KeyError:
                    continue

                distance = np.linalg.norm(pos_i - pos_j)
                seq_separation = abs(res_id_i - res_id_j)

                if distance <= threshold:
                     # Determine interaction features
                    hydrophobic_interaction = 1 if aa_i in ['A', 'I', 'L', 'V', 'F', 'M'] and aa_j in ['A', 'I', 'L', 'V', 'F', 'M'] else 0
                    polar_interaction = 1 if aa_i in ['R', 'N', 'D', 'E', 'Q', 'H', 'K', 'S', 'T', 'Y'] and aa_j in ['R', 'N', 'D', 'E', 'Q', 'H', 'K', 'S', 'T', 'Y'] else 0
                    charge_compatibility = 1 if (aa_i in ['R', 'K', 'H'] and aa_j in ['D', 'E']) or (aa_j in ['R', 'K', 'H'] and aa_i in ['D', 'E']) else 0
                    covalent_bond = 1 if aa_i == 'C' and aa_j == 'C' and distance < 2.5 else 0  # Approximate distance for disulfide bonds

                    # New features
                    aromatic_interaction = 1 if aa_i in ['F', 'Y', 'W'] and aa_j in ['F', 'Y', 'W'] else 0
                    hydrogen_bond_potential = 1 if (aa_i in ['S', 'T', 'N', 'Q'] and aa_j in ['S', 'T', 'N', 'Q']) or (aa_i in ['R', 'K', 'H'] and aa_j in ['D', 'E', 'N', 'Q']) or (aa_j in ['R', 'K', 'H'] and aa_i in ['D', 'E', 'N', 'Q']) else 0
                    size_compatibility = 1 if (aa_i in ['A', 'G', 'S'] and aa_j in ['F', 'W', 'Y']) or (aa_j in ['A', 'G', 'S'] and aa_i in ['F', 'W', 'Y']) else 0
                    proline_interaction = 1 if aa_i == 'P' or aa_j == 'P' else 0

                    # Combine all edge features
                    edge_features = np.array([
                        distance, 
                        seq_separation, 
                        hydrophobic_interaction, 
                        polar_interaction, 
                        charge_compatibility, 
                        covalent_bond,
                        aromatic_interaction,
                        hydrogen_bond_potential,
                        size_compatibility,
                        proline_interaction
                    ])

                    edge_type = (f"{aa_i}_protein", 'contact', f"{aa_j}_protein") if aa_i <= aa_j else (f"{aa_j}_protein", 'contact', f"{aa_i}_protein")
                    src_aa, tgt_aa = edge_type[0].split('_')[0], edge_type[2].split('_')[0]
                    src_global, tgt_global = (i, j) if aa_i <= aa_j else (j, i)

                    if edge_type not in contact_edge_index:
                        contact_edge_index[edge_type] = {'edge_index': [], 'edge_attr': []}

                    src_local = global_to_local_idx[src_aa][src_global]
                    tgt_local = global_to_local_idx[tgt_aa][tgt_global]

                    # Append edge index and attributes
                    contact_edge_index[edge_type]['edge_index'].append([src_local, tgt_local])
                    contact_edge_index[edge_type]['edge_attr'].append(edge_features)
                    edge_types.add(edge_type)

        # Add edges and their reverse to HeteroData object
        for edge_type, attrs in contact_edge_index.items():
            if attrs['edge_index']:
                src_type, relation, tgt_type = edge_type
                reverse_edge_type = (tgt_type, relation, src_type)
                reverse_edge_types.add(reverse_edge_type)

                # Original edges
                edge_tensor = torch.tensor(attrs['edge_index'], dtype=torch.long).t().contiguous()
                edge_attr_tensor = torch.tensor(attrs['edge_attr'], dtype=torch.float)
                data[edge_type].edge_index = edge_tensor
                data[edge_type].edge_attr = edge_attr_tensor

                # Reverse edges
                reverse_edge_index = torch.tensor([[tgt, src] for src, tgt in attrs['edge_index']], dtype=torch.long).t().contiguous()
                reverse_edge_attr = edge_attr_tensor  # Edge attributes remain the same for reverse edges
                data[reverse_edge_type].edge_index = reverse_edge_index
                data[reverse_edge_type].edge_attr = reverse_edge_attr

        # Add metadata to HeteroData object
        data.node_types = set(f"{aa}_protein" for aa in unique_amino_acids)
        data.edge_types = edge_types.union(reverse_edge_types)
        data.reverse_edge_types = reverse_edge_types

        return data