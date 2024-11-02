# Third-party imports
import pandas as pd
import torch
from torch_geometric.data import Dataset

# Import MoleculeDataset from molecule module
from .molecule import MoleculeDataset

class CombinedDataset(Dataset):
    def __init__(self, dataframe, transform=None, pre_transform=None, pre_filter=None, predicting=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.protein_mapping = {
            "HSA": 0,
            "sEH": 1,
            "BRD4": 2
        }
        self.predicting = predicting
        super(CombinedDataset, self).__init__(None, transform, pre_transform, pre_filter)

    def len(self):
        return len(self.dataframe)

    def get(self, idx):
        row = self.dataframe.iloc[idx]
        protein_name = row['protein_name']

        mol_dataset = MoleculeDataset(pd.DataFrame([row]))
        
        if self.predicting:
            node = mol_dataset.get(0, predicting=True)
            if node['invalid']:
                return None
                
            protein_type_id = torch.tensor([self.protein_mapping[protein_name]], dtype=torch.long)
            return node, protein_type_id
            
        anchor, positive, negative = mol_dataset.get(0)

        if any(graph['invalid'] for graph in [anchor, positive, negative]):
            return None

        protein_type_id = self.protein_mapping[protein_name]

        # Convert protein_type_id to a tensor 
        protein_type_id = torch.tensor([protein_type_id], dtype=torch.long)

        return anchor, positive, negative, protein_type_id
