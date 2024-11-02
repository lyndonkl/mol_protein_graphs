
# Standard library imports
import os

# Third-party imports
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Dataset, HeteroData
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MoleculeDataset(Dataset):
    def __init__(self, dataframe, transform=None, pre_transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        super(MoleculeDataset, self).__init__(None, transform, pre_transform)

    def len(self):
        return len(self.dataframe)

    def get(self, idx, predicting=False):
        row = self.dataframe.iloc[idx]
        protein_name = row['protein_name']
        molecule_id = row['id']

        if predicting:
            molecule_smiles = row['molecule_smiles']
            node = self.create_molecule_graph(molecule_smiles)
            node['smolecule'].protein_name = protein_name
            node['smolecule'].id = str(molecule_id)
            return node

        smiles_binds = row['smiles_binds']
        smiles_non_binds_1 = row['smiles_non_binds_1']
        smiles_non_binds_2 = row['smiles_non_binds_2']

        # Create three molecule graphs
        anchor = self.create_molecule_graph(smiles_non_binds_1)
        positive = self.create_molecule_graph(smiles_non_binds_2)
        negative = self.create_molecule_graph(smiles_binds)

        # Add metadata
        for graph in [anchor, positive, negative]:
            graph['smolecule'].protein_name = protein_name
            # Store molecule_id as a string to avoid overflow issues
            graph['smolecule'].id = str(molecule_id)

        return anchor, positive, negative

    def create_molecule_graph(self, smiles):
        # Convert SMILES to molecular graph
        mol = Chem.MolFromSmiles(smiles)
        data = HeteroData()
        data['invalid'] = False
        
        if mol is None:
            # Handle invalid SMILES
            data['invalid'] = True
            data['dummy_node'].x = torch.zeros((1, 1), dtype=torch.float)  # Dummy node feature
            data['dummy_node'].y = torch.tensor([0], dtype=torch.float)  # Dummy label
            data['dummy_node', 'to', 'dummy_node'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop edge
            data['dummy_node', 'to', 'dummy_node'].edge_attr = torch.zeros((1, 1), dtype=torch.float)  # Dummy edge attribute
            
            # Update node and edge types
            data.edge_types = [('dummy_node', 'to', 'dummy_node')]
            data.node_types = ['dummy_node'] 
            return data

        # Remove Dysprosium atoms if present
        atoms_to_remove = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'Dy']
        mol = Chem.EditableMol(mol)
        for idx in sorted(atoms_to_remove, reverse=True):
            mol.RemoveAtom(idx)
        mol = mol.GetMol()

        mol = Chem.AddHs(mol)
        
        # Embed molecule
        AllChem.EmbedMolecule(mol, randomSeed=42)

        atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
        unique_atom_types = list(set(atom_types))

        atom_type_to_indices = {atype: [] for atype in unique_atom_types}
        atom_features = []
        atom_positions = []
        conformer = None

        try:
            conformer = mol.GetConformer()
        except Exception as e:
            print(f"Skipping molecule due to error: {e}")
            data['invalid'] = True
            return data

        if conformer is None:
            # Handle invalid conformer
            data['invalid'] = True
            data['dummy_node'].x = torch.zeros((1, 1), dtype=torch.float)  # Dummy node feature
            data['dummy_node'].y = torch.tensor([0], dtype=torch.float)  # Dummy label
            data['dummy_node', 'to', 'dummy_node'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop edge
            data['dummy_node', 'to', 'dummy_node'].edge_attr = torch.zeros((1, 1), dtype=torch.float)  # Dummy edge attribute
            
            # Update node and edge types
            data.edge_types = [('dummy_node', 'to', 'dummy_node')]
            data.node_types = ['dummy_node'] 
            return data

        for i, atom in enumerate(mol.GetAtoms()):
            atype = atom.GetSymbol()
            atom_type_to_indices[atype].append(i)
            atom_features.append(self.get_atom_features(atom))
            
            pos = mol.GetConformer().GetAtomPosition(i)
            atom_positions.append(np.array([pos.x, pos.y, pos.z], dtype=np.float32))

        # Assign node features and positions per atom type
        for atype in unique_atom_types:
            idx = atom_type_to_indices[atype]
            feature_list = np.array([atom_features[i] for i in idx])
            x_feats = torch.tensor(feature_list, dtype=torch.float)
            pos_list = np.array([atom_positions[i] for i in idx])
            pos = torch.tensor(pos_list, dtype=torch.float)
            # Concatenate atom features with positions
            x = torch.cat([x_feats, pos], dim=1)
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
        reverse_edge_types = set()
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            atype_i = atom_types[i]
            atype_j = atom_types[j]
            
            # Define edge type in consistent order
            if atype_i <= atype_j:
                edge_type = (atype_i, 'bond', atype_j)
                reverse_edge_type = (atype_j, 'bond', atype_i)
                src_atype, tgt_atype = atype_i, atype_j
                src_idx, tgt_idx = i, j
            else:
                edge_type = (atype_j, 'bond', atype_i)
                reverse_edge_type = (atype_i, 'bond', atype_j)
                src_atype, tgt_atype = atype_j, atype_i
                src_idx, tgt_idx = j, i
            edge_types.add(edge_type)
            reverse_edge_types.add(reverse_edge_type)

            if edge_type not in bond_edges:
                bond_edges[edge_type] = {'edge_index': [], 'edge_attr': []}

            if reverse_edge_type not in bond_edges:
                bond_edges[reverse_edge_type] = {'edge_index': [], 'edge_attr': []}

            # Retrieve local indices using precomputed mapping
            src_local = atom_type_to_local_idx[src_atype][src_idx]
            tgt_local = atom_type_to_local_idx[tgt_atype][tgt_idx]

            # Append both directions for undirected bonds
            bond_edges[edge_type]['edge_index'].append([src_local, tgt_local])
            bond_edges[reverse_edge_type]['edge_index'].append([tgt_local, src_local])

            # Append bond features for both directions
            bond_feature = self.get_bond_features(bond)
            bond_edges[edge_type]['edge_attr'].append(bond_feature)
            bond_edges[reverse_edge_type]['edge_attr'].append(bond_feature)

        # Assign bond edges to HeteroData
        for edge_type, attrs in bond_edges.items():
            edge_index = torch.tensor(attrs['edge_index'], dtype=torch.long).t().contiguous()
            edge_attrs_list = np.array(attrs['edge_attr'])
            edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float)
            data[edge_type].edge_index = edge_index
            data[edge_type].edge_attr = edge_attr

        data['smolecule'].smiles = smiles

        data.node_types = set(unique_atom_types)
        data.edge_types = set(bond_edges.keys())

        return data

    @staticmethod
    def get_atom_features(atom):
        atomic_number = atom.GetAtomicNum()
        mol = atom.GetOwningMol()

        # Pauling electronegativity values for common elements
        electronegativity = {
            1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16,
            35: 2.96, 53: 2.66
        }
        
        features = [
            atomic_number,
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            int(atom.GetIsAromatic()),
            Chem.GetPeriodicTable().GetRvdw(atomic_number),
            electronegativity.get(atomic_number, 0), 
            Chem.GetPeriodicTable().GetNOuterElecs(atomic_number)
        ]
        
        # Return as a numpy array
        return np.array(features, dtype=np.float32)

    @staticmethod
    def get_bond_features(bond):
        mol = bond.GetOwningMol()
        bond_type = bond.GetBondType()
        bond_dict = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3
        }

        # Calculate bond length manually
        conf = mol.GetConformer()
        a1 = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        a2 = conf.GetAtomPosition(bond.GetEndAtomIdx())
        bond_length = np.linalg.norm(np.array([a1.x - a2.x, a1.y - a2.y, a1.z - a2.z]))

        features = [
            bond_dict.get(bond_type, -1),
            bond_length,
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetIsAromatic()),
            bond.GetValenceContrib(bond.GetBeginAtom()),
            bond.GetValenceContrib(bond.GetEndAtom()),
        ]

        # Stereo configuration
        stereo = bond.GetStereo()
        stereo_dict = {
            Chem.rdchem.BondStereo.STEREONONE: 0,
            Chem.rdchem.BondStereo.STEREOANY: 1,
            Chem.rdchem.BondStereo.STEREOZ: 2,
            Chem.rdchem.BondStereo.STEREOE: 3,
        }
        features.append(stereo_dict.get(stereo, -1))

        # Bond angles
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if begin_atom.GetDegree() > 1 and end_atom.GetDegree() > 1:
            conf = mol.GetConformer()
            begin_neighbors = [a.GetIdx() for a in begin_atom.GetNeighbors() if a.GetIdx() != end_atom.GetIdx()]
            end_neighbors = [a.GetIdx() for a in end_atom.GetNeighbors() if a.GetIdx() != begin_atom.GetIdx()]
            begin_angle = Chem.rdMolTransforms.GetAngleDeg(conf, begin_neighbors[0], begin_atom.GetIdx(), end_atom.GetIdx())
            end_angle = Chem.rdMolTransforms.GetAngleDeg(conf, begin_atom.GetIdx(), end_atom.GetIdx(), end_neighbors[0])
            features.extend([begin_angle, end_angle])
        else:
            features.extend([0, 0])  # Placeholder for molecules without angles

        return np.array(features, dtype=np.float32)
