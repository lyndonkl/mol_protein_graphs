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
            # Since the graph is undirected, add reverse edges
            reverse_edges = [[tgt, src] for src, tgt in edges]
            all_edges = edges + reverse_edges
            edge_tensor = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
            data[edge_type].edge_index = edge_tensor

    data.node_types = set(unique_amino_acids)
    data.edge_types = edge_types

    return data

class MoleculeDataset(Dataset):
    def __init__(self, dataframe, transform=None, pre_transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        super(MoleculeDataset, self).__init__(None, transform, pre_transform)

    def len(self):
        return len(self.dataframe)

    def get(self, idx):
        row = self.dataframe.iloc[idx]
        smiles = row['molecule_smiles']
        binds = row['binds']
        protein_name = row['protein_name']

        # Convert SMILES to molecular graph
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None  # Skip invalid SMILES

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
        unique_atom_types = list(set(atom_types))

        data = HeteroData()

        atom_type_to_indices = {atype: [] for atype in unique_atom_types}
        atom_features = []
        atom_positions = []
        for i, atom in enumerate(mol.GetAtoms()):
            atype = atom.GetSymbol()
            atom_type_to_indices[atype].append(i)
            atom_features.append(self.get_atom_features(atom))
            pos = mol.GetConformer().GetAtomPosition(i)
            atom_positions.append(np.array([pos.x, pos.y, pos.z], dtype=np.float32))

        # Assign node features and positions per atom type
        for atype in unique_atom_types:
            idx = atom_type_to_indices[atype]
            x = torch.tensor([atom_features[i] for i in idx], dtype=torch.float)
            pos = torch.tensor(np.array([atom_positions[i] for i in idx]), dtype=torch.float)
            data[atype].x = x
            data[atype].pos = pos

        # Precompute mapping from global atom index to local index within atom type
        atom_type_to_local_idx = {
            atype: {global_idx: local_idx for local_idx, global_idx in enumerate(idxs)}
            for atype, idxs in atom_type_to_indices.items()
        }

        # Assign bond edges to specific edge types based on atom types
        bond_edges = {}
        edge_types = set()
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            atype_i = atom_types[i]
            atype_j = atom_types[j]
            
            # Define edge type in consistent order
            if atype_i <= atype_j:
                edge_type = (atype_i, 'bond', atype_j)
                src_atype, tgt_atype = atype_i, atype_j
                src_idx, tgt_idx = i, j
            else:
                edge_type = (atype_j, 'bond', atype_i)
                src_atype, tgt_atype = atype_j, atype_i
                src_idx, tgt_idx = j, i
            edge_types.add(edge_type)
            if edge_type not in bond_edges:
                bond_edges[edge_type] = {'edge_index': [], 'edge_attr': []}

            # Retrieve local indices using precomputed mapping
            src_local = atom_type_to_local_idx[src_atype][src_idx]
            tgt_local = atom_type_to_local_idx[tgt_atype][tgt_idx]

            # Append both directions for undirected bonds
            bond_edges[edge_type]['edge_index'].append([src_local, tgt_local])
            bond_edges[edge_type]['edge_index'].append([tgt_local, src_local])

            # Append bond features for both directions
            bond_feature = self.get_bond_features(bond)
            bond_edges[edge_type]['edge_attr'].append(bond_feature)
            bond_edges[edge_type]['edge_attr'].append(bond_feature)

        # Assign bond edges to HeteroData
        for edge_type, attrs in bond_edges.items():
            edge_index = torch.tensor(attrs['edge_index'], dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(attrs['edge_attr'], dtype=torch.float)
            data[edge_type].edge_index = edge_index
            data[edge_type].edge_attr = edge_attr

        # Add binding label and metadata
        data['smolecule'].y = torch.tensor([binds], dtype=torch.float)
        data['smolecule'].smiles = smiles
        data['smolecule'].protein_name = protein_name

        data.node_types = set(unique_atom_types)
        data.edge_types = set(bond_edges.keys())

        return data

    @staticmethod
    def get_atom_features(atom):
        return [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            int(atom.GetIsAromatic())
        ]

    @staticmethod
    def get_bond_features(bond):
        bond_type = bond.GetBondType()
        bond_dict = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3
        }

        return [
            bond_dict.get(bond_type, -1)
        ]


class CombinedDataset(Dataset):
    def __init__(self, dataframe, protein_graphs, transform=None, pre_transform=None, cache_dir='./processed'):
        self.dataframe = dataframe.reset_index(drop=True)
        self.protein_graphs = protein_graphs
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        super(CombinedDataset, self).__init__(None, transform, pre_transform)

    def len(self):
        return len(self.dataframe)

    def get(self, idx):
        processed_file = os.path.join(self.cache_dir, f'data_{idx}.pt')
        if os.path.exists(processed_file):
            molecule_data, protein_data = torch.load(processed_file)
        else:
            row = self.dataframe.iloc[idx]
            smiles = row['molecule_smiles']
            binds = row['binds']
            protein_name = row['protein_name']

            mol_dataset = MoleculeDataset(pd.DataFrame([row]))
            molecule_data = mol_dataset.get(0)
            if molecule_data is None:
                return None

            protein_data = self.protein_graphs.get(protein_name, HeteroData())

            molecule_data.y = torch.tensor([binds], dtype=torch.float)
            molecule_data.smiles = smiles
            molecule_data.protein_name = protein_name

            torch.save((molecule_data, protein_data), processed_file)

        return molecule_data, protein_data