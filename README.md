# leash-belka
Predicting Small Molecule-Protein Binding Using Graph Neural Networks

# Predicting Small Molecule-Protein Binding Using Graph Neural Networks

## Project Overview
This project aims to predict whether small molecules bind to one of three proteins using Graph Neural Networks (GNNs). Each small molecule is composed of three building blocks, and the dataset includes information in the form of SMILES strings for the building blocks, the small molecules, and crystal PDB structures for the proteins. Additionally, the dataset includes a binary label indicating whether each small molecule binds to a given protein.

## Table of Contents
- [Project Overview](#project-overview)
- [Challenges](#challenges)
- [Approach](#approach)
- [Implementation Steps](#implementation-steps)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Challenges
1. **Variety in Small Molecules:** The same building blocks can result in different small molecules, some of which bind to a given protein while others do not.
2. **Unseen Small Molecules:** The model needs to generalize to small molecules that were not seen during training, even if the building blocks were seen.
3. **Combining Features:** Effectively combining features from building blocks and small molecules to make accurate binding predictions.

## Approach
1. **Graph Representation:**
    - **Nodes:**
        - **Building Blocks:** Represent the individual building blocks using features extracted from SMILES data.
        - **Small Molecules:** Represent the small molecules using features extracted from SMILES data.
        - **Proteins:** Represent the proteins using features extracted from PDB structures.
    - **Edges:**
        - **Composition Edges:** Connect building blocks to the small molecules they compose.
        - **Binding Edges:** Connect small molecules to proteins if they bind to the protein.
2. **Feature Extraction:**
    - **RDKit:** Extract molecular descriptors and fingerprints from SMILES data for building blocks and small molecules.
    - **Biopython:** Extract structural features from PDB data for proteins.
3. **Model:**
    - Use Graph Convolutional Transformer (GCT) layers with attention mechanisms to capture complex relationships and interactions within the graph.
    - Train the model using neighborhood sampling to handle large graphs efficiently.
4. **Prediction:**
    - Combine features from building blocks and small molecules to predict binding probabilities.
    - Handle unseen small molecules by leveraging known building block features.

## Implementation Steps
1. **Feature Extraction:** Use RDKit and Biopython to extract relevant features from SMILES and PDB data.
2. **Graph Construction:** Build a heterogeneous graph in DGL with nodes and edges representing the building blocks, small molecules, and proteins.
3. **Model Definition:** Define a GCN-based model with transformer layers to perform link prediction.
4. **Neighborhood Sampling:** Use DGL's neighborhood sampler and edge data loader for efficient minibatch training.
5. **Training Loop:** Train the model using the defined neighborhood sampling strategy.
6. **Prediction:** Combine building block and small molecule features to make binding predictions for unseen data.

## Setup Instructions
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/small-molecule-protein-binding.git
    cd small-molecule-protein-binding
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv env
    source env/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. **Feature Extraction:**
    Extract features using RDKit and Biopython:
    ```python
    from feature_extraction import extract_features
    pdb_file = "path/to/protein.pdb"
    ligand_smiles = "CCO"
    features = extract_features(pdb_file, ligand_smiles)
    ```

2. **Graph Construction:**
    Build the heterogeneous graph:
    ```python
    from graph_construction import build_graph
    g = build_graph(train_data)
    ```

3. **Model Training:**
    Train the GNN model:
    ```python
    from model_training import train_model
    train_model(g)
    ```

4. **Prediction:**
    Predict binding probabilities for unseen data:
    ```python
    from prediction import predict_binding
    prediction = predict_binding(unseen_smiles, building_block_smiles, protein_id)
    ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any bugs, features, or improvements.

