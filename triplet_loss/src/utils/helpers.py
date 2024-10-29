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

def collate_fn(batch):
    """
    Collate function to process and combine batch items for triplet loss.
    """
    valid_items = [item for item in batch if item is not None]
    
    anchors = [item[0] for item in valid_items]
    positives = [item[1] for item in valid_items]
    negatives = [item[2] for item in valid_items]
    protein_type_ids = torch.cat([item[3] for item in valid_items])

    anchor_batch = Batch.from_data_list(anchors)
    positive_batch = Batch.from_data_list(positives)
    negative_batch = Batch.from_data_list(negatives)

    batch_size = len(valid_items)

    return anchor_batch, positive_batch, negative_batch, protein_type_ids, batch_size
