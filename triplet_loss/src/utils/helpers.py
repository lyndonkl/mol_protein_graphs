import logging
from torch_geometric.data import Batch
import torch

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

def collate_fn(batch, predicting=False):
    """
    Collate function to process and combine batch items for triplet loss.
    If predicting=True, returns only the first item (node) and protein_type_id.
    Otherwise returns the full triplet data.
    """
    valid_items = [item for item in batch if item is not None]
    
    if predicting:
        nodes = [item[0] for item in valid_items]
        protein_type_ids = torch.cat([item[1] for item in valid_items])
        combined_data = Batch.from_data_list(nodes)
        return combined_data, protein_type_ids, len(valid_items), len(valid_items)
    
    anchors = [item[0] for item in valid_items]
    positives = [item[1] for item in valid_items]
    negatives = [item[2] for item in valid_items]
    protein_type_ids = torch.cat([item[3] for item in valid_items])

    num_items = len(valid_items)
    batch_size = num_items * 3

    # Stack anchor, positive, and negative, and repeat protein_type_ids
    combined_data = Batch.from_data_list(anchors + positives + negatives)
    combined_protein_id = torch.cat([protein_type_ids, protein_type_ids, protein_type_ids], dim=0)

    return combined_data, combined_protein_id, batch_size, num_items
