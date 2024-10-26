import os
import logging
from itertools import cycle
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pyarrow.parquet as pq
import pyarrow as pa
from rdkit import Chem
from rdkit.Chem import AllChem
import concurrent.futures
from functools import partial
import random

class Preprocessor:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir
        self.proteins = ['HSA', 'sEH', 'BRD4']
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def check_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            atoms_to_remove = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'Dy']
            mol = Chem.EditableMol(mol)
            for idx in sorted(atoms_to_remove, reverse=True):
                mol.RemoveAtom(idx)
            mol = mol.GetMol()

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)

            try:
                conformer = mol.GetConformer()
            except Exception:
                return False

            if conformer is None:
                return False

            return True
        except Exception:
            return False

    def process_chunk(self, chunk, chunk_id):
        self.logger.info(f"Processing chunk {chunk_id}")
        for protein in self.proteins:
            df_protein = chunk[chunk['protein_name'] == protein]
            df_binds = df_protein[df_protein['binds'] == 1]
            df_non_binds = df_protein[df_protein['binds'] == 0]

            if not df_binds.empty:
                binds_file = f"{self.output_dir}/{protein}_binds_chunk_{chunk_id}.parquet"
                df_binds.to_parquet(binds_file, engine='pyarrow', compression='snappy')

            if not df_non_binds.empty:
                non_binds_file = f"{self.output_dir}/{protein}_non_binds_chunk_{chunk_id}.parquet"
                df_non_binds.to_parquet(non_binds_file, engine='pyarrow', compression='snappy')
        self.logger.info(f"Finished processing chunk {chunk_id}")

    def process_bind_chunk(self, bind_chunk, non_binds_files, protein):
        result_rows = []
        non_binds_file_iter = iter(non_binds_files)
        current_non_binds = pd.read_parquet(next(non_binds_file_iter), engine='pyarrow')
        non_binds_row_iter = iter(current_non_binds.iterrows())

        for _, bind_row in bind_chunk.iterrows():
            if not self.check_smiles(bind_row['molecule_smiles']):
                continue

            non_binds = []
            while len(non_binds) < 2:
                try:
                    _, non_bind = next(non_binds_row_iter)
                except StopIteration:
                    try:
                        current_non_binds = pd.read_parquet(next(non_binds_file_iter), engine='pyarrow')
                        non_binds_row_iter = iter(current_non_binds.iterrows())
                        continue
                    except StopIteration:
                        break

                if self.check_smiles(non_bind['molecule_smiles']):
                    non_binds.append(non_bind)

            if len(non_binds) < 2:
                break

            result_rows.append({
                'id': f"{bind_row['id']}_{non_binds[0]['id']}_{non_binds[1]['id']}",
                'smiles_binds': bind_row['molecule_smiles'],
                'smiles_non_binds_1': non_binds[0]['molecule_smiles'],
                'smiles_non_binds_2': non_binds[1]['molecule_smiles'],
                'protein_name': bind_row['protein_name']
            })

        return result_rows

    def merge_and_sample_non_binds(self, protein):
        self.logger.info(f"Starting merge and sample for protein {protein}")
        binds_files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) if f.startswith(f"{protein}_binds_chunk_")]
        non_binds_files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) if f.startswith(f"{protein}_non_binds_chunk_")]

        if not binds_files or not non_binds_files:
            self.logger.warning(f"No files found for protein {protein}")
            return None

        binds_data = pd.concat(pd.read_parquet(file, engine='pyarrow') for file in binds_files)
        max_binds = len(binds_data) * 3

        chunk_size = 1000
        binds_chunks = [binds_data[i:i+chunk_size] for i in range(0, len(binds_data), chunk_size)]

        output_file = f"{self.output_dir}/{protein}_cleaned.parquet"

        schema = pa.schema([
            ('id', pa.string()),
            ('smiles_binds', pa.string()),
            ('smiles_non_binds_1', pa.string()),
            ('smiles_non_binds_2', pa.string()),
            ('protein_name', pa.string())
        ])

        with pq.ParquetWriter(output_file, schema) as writer:
            self.logger.info(f"Processing {max_binds} binds for protein {protein}")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for bind_chunk in binds_chunks:
                    non_binds_files_subset = random.sample(non_binds_files, min(len(non_binds_files), 10))
                    future = executor.submit(self.process_bind_chunk, bind_chunk, non_binds_files_subset, protein)
                    futures.append(future)
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {protein}"):
                    chunk_results = future.result()
                    if chunk_results:
                        table = pa.Table.from_pylist(chunk_results, schema=schema)
                        writer.write_table(table)

        self.logger.info(f"Finished processing for protein {protein}. Cleaning up intermediate files.")
        for f in binds_files + non_binds_files:
            os.remove(f)

    def process(self):
        self.logger.info(f"Starting to process large parquet file: {self.input_file}")
        parquet_file = pq.ParquetFile(self.input_file)
        
        tasks = []
        with tqdm(total=parquet_file.num_row_groups, dynamic_ncols=True) as pbar:
            for i in range(parquet_file.num_row_groups):
                chunk = parquet_file.read_row_group(i).to_pandas()
                tasks.append(delayed(self.process_chunk)(chunk, i))
                pbar.update(1)

        self.logger.info("Executing parallel tasks for chunk processing")
        Parallel(n_jobs=-1, timeout=600)(tasks)

        self.logger.info("Starting merge and sample for each protein")
        for protein in self.proteins:
            self.merge_and_sample_non_binds(protein)

        self.logger.info("Merging final cleaned data for all proteins")
        cleaned_files = [f"{self.output_dir}/{protein}_cleaned.parquet" for protein in self.proteins]
        df_final = pd.concat([pd.read_parquet(f) for f in cleaned_files])
        df_final.to_parquet(f"{self.output_dir}/cleaned_train.parquet", engine='pyarrow', compression='snappy')

        self.logger.info("Cleaning up intermediate files")
        for f in cleaned_files:
            os.remove(f)

        self.logger.info("Processing complete")

    def remove_duplicate_ids(self):
        self.logger.info("Removing duplicate IDs from cleaned_train.parquet")
        df = pd.read_parquet(f"{self.output_dir}/cleaned_train.parquet")
        df_unique = df.drop_duplicates(subset='id', keep='first')
        df_unique.to_parquet(f"{self.output_dir}/cleaned_train_unique.parquet", engine='pyarrow', compression='snappy')
        self.logger.info(f"Removed {len(df) - len(df_unique)} duplicate rows")

    def sample_and_verify(self, sample_size=500000):
        self.logger.info(f"Sampling {sample_size} rows and verifying against original data")
        df_cleaned = pd.read_parquet(f"{self.output_dir}/cleaned_train_unique.parquet")

        # Sample rows
        df_sample = df_cleaned.sample(n=min(sample_size, len(df_cleaned)), random_state=42)

        correct_count = 0
        pf = pq.ParquetFile(self.input_file)
        num_row_groups = pf.num_row_groups

        # Create a dictionary to store the IDs we need to verify
        ids_to_verify = {}
        for _, row in df_sample.iterrows():
            for id in row['id'].split('_'):
                ids_to_verify[id] = row['id']

        total_ids = len(ids_to_verify)
        verified_ids = set()
        with tqdm(total=total_ids, desc="Verifying IDs") as pbar:
            for i in tqdm(range(num_row_groups), desc="Reading original data"):
                print(f"Processing row group {i}")
                chunk = pf.read_row_group(i).to_pandas()
                
                # Filter the chunk to only include rows we need to verify
                chunk = chunk[chunk['id'].astype(str).isin(ids_to_verify.keys())]
                
                if not chunk.empty:
                    chunk_dict = chunk.set_index('id').to_dict('index')
                    
                    for orig_id, orig_data in chunk_dict.items():
                        orig_id = str(orig_id)
                        if orig_id in ids_to_verify:
                            sample_row = df_sample[df_sample['id'] == ids_to_verify[orig_id]].iloc[0]
                            
                            sample_id_parts = ids_to_verify[orig_id].split('_')
                            position_correct = False
                            smiles_correct = False

                            if orig_data['binds'] == 1:
                                position_correct = sample_id_parts[0] == orig_id
                                smiles_correct = sample_row['smiles_binds'] == orig_data['molecule_smiles']
                            elif orig_data['binds'] == 0:
                                if sample_id_parts[1] == orig_id:
                                    position_correct = True
                                    smiles_correct = sample_row['smiles_non_binds_1'] == orig_data['molecule_smiles']
                                elif sample_id_parts[2] == orig_id:
                                    position_correct = True
                                    smiles_correct = sample_row['smiles_non_binds_2'] == orig_data['molecule_smiles']

                            protein_correct = sample_row['protein_name'] == orig_data['protein_name']

                            if position_correct and smiles_correct and protein_correct:
                                correct_count += 1
                            else:
                                self.logger.warning(f"Mismatch found for ID {orig_id}:")
                                self.logger.warning(f"Position correct: {position_correct}")
                                self.logger.warning(f"SMILES correct: {smiles_correct}")
                                self.logger.warning(f"Protein correct: {protein_correct}")
                                self.logger.warning(f"Original data: {orig_data}")
                                self.logger.warning(f"Sample row: {sample_row.to_dict()}")

                            # Add verified ID to the set
                            verified_ids.add(orig_id)
                            # Remove verified ID from the dictionary
                            del ids_to_verify[orig_id]
                            pbar.update(1)
                
                # If all IDs are verified, break the loop
                if not ids_to_verify:
                    break

        total_verified = len(verified_ids)
        accuracy = correct_count / total_verified
        self.logger.info(f"Verification accuracy: {accuracy:.2%}")
        self.logger.info(f"Total number of unique IDs verified: {total_verified}")
        total_rows = len(df_cleaned)
        self.logger.info(f"Total number of rows in the dataset: {total_rows}")

    def print_dataset_statistics(self):
        self.logger.info("Calculating dataset statistics")
        df = pd.read_parquet(f"{self.output_dir}/cleaned_train_unique.parquet")
        
        # Count unique SMILES
        unique_binding_smiles = df['smiles_binds'].nunique()
        unique_non_binding_smiles_1 = df['smiles_non_binds_1'].nunique()
        unique_non_binding_smiles_2 = df['smiles_non_binds_2'].nunique()
        total_unique_non_binding_smiles = pd.concat([df['smiles_non_binds_1'], df['smiles_non_binds_2']]).nunique()
        
        # Count triplets per protein
        triplets_per_protein = df['protein_name'].value_counts()
        
        # Print statistics
        self.logger.info(f"Total number of triplets: {len(df)}")
        self.logger.info(f"Number of unique binding SMILES: {unique_binding_smiles}")
        self.logger.info(f"Number of unique non-binding SMILES: {total_unique_non_binding_smiles}")
        self.logger.info(f"Number of unique non-binding SMILES (position 1): {unique_non_binding_smiles_1}")
        self.logger.info(f"Number of unique non-binding SMILES (position 2): {unique_non_binding_smiles_2}")
        self.logger.info("\nTriplets per protein:")
        for protein, count in triplets_per_protein.items():
            self.logger.info(f"  {protein}: {count}")

if __name__ == "__main__":
    input_parquet = "train.parquet"
    output_directory = "."
    preprocessor = Preprocessor(input_parquet, output_directory)
    # preprocessor.process()
    # preprocessor.remove_duplicate_ids()
    # preprocessor.sample_and_verify()
    preprocessor.print_dataset_statistics()
