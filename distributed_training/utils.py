# Standard library imports
import logging

# Third-party imports
from torch_geometric.data import Batch

def custom_transform(batch):
    """
    Custom transform function to prepare batch data for model input.
    """
    mol_batch, prot_batch, batch_size = collate_fn(batch)
    return {
        'mol_batch': mol_batch,
        'prot_batch': prot_batch,
        'batch_size': batch_size
    }

def collate_fn(batch):
    """
    Collate function to process and combine batch items.
    """
    valid_items = [item for item in batch if item is not None and item[0] is not None and item[0]['invalid'] is False]
    
    mol_batch = [item[0] for item in valid_items]
    prot_batch = [item[1] for item in valid_items]

    mol_batch = Batch.from_data_list(mol_batch)
    prot_batch = Batch.from_data_list(prot_batch)

    batch_size = len(valid_items)

    return mol_batch, prot_batch, batch_size

def collect_protein_node_and_edge_types(protein_graphs):
    """
    Collect unique protein node and edge types from protein graphs.
    """
    protein_node_types = set()
    protein_edge_types = set()
    for protein_data in protein_graphs.values():
        protein_node_types.update(protein_data.node_types)
        protein_edge_types.update(protein_data.edge_types)
        protein_edge_types.update(protein_data.reverse_edge_types)
    return sorted(protein_node_types), sorted(protein_edge_types)

def setup_logger():
    """
    Set up and configure a logger for training.
    """
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger