def custom_transform(batch):
    mol_batch, prot_batch, batch_size = collate_fn(batch)
    return {
        'mol_batch': mol_batch,
        'prot_batch': prot_batch,
        'batch_size': batch_size
    }

def collate_fn(batch):
    valid_items = [item for item in batch if item is not None and item[0] is not None and item[0]['invalid'] is False]
    
    mol_batch = [item[0] for item in valid_items]
    prot_batch = [item[1] for item in valid_items]

    mol_batch = Batch.from_data_list(mol_batch)
    prot_batch = Batch.from_data_list(prot_batch)

    batch_size = len(valid_items)

    return mol_batch, prot_batch, batch_size

def setup_logger():
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger