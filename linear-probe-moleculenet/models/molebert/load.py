import sys
import json

from pathlib import Path, PosixPath

import torch

try:
    from .model import GNN_graphpred
    from .tokenizer import smi_to_graph_data_obj_simple
except ImportError:
    from model import GNN_graphpred
    from tokenizer import smi_to_graph_data_obj_simple


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/Mole-BERT'


def load_molebert(path: PosixPath = model_path):
    with open(path / 'config.json', 'r') as f:
        config = json.load(f)
        
    model = GNN_graphpred(**config)
    model.from_pretrained(path / 'Mole-BERT.pth')

    return model


def load_molebert_tokenizer(path: PosixPath = model_path):
    return smi_to_graph_data_obj_simple


if __name__ == '__main__':
    tokenizer = load_molebert_tokenizer()
    model = load_molebert()

    smiles = '*C(=O)[C@H](CCCCNC(=O)OCCOC)NC(=O)OCCOC'
    from rdkit.Chem import AllChem
    mol = AllChem.MolFromSmiles(smiles)

    if mol is not None:
        print("Molecule created successfully!")
    else:
        print("Invalid SMILES string.")

    output = model(tokenizer('*C(=O)[C@H](CCCCNC(=O)OCCOC)NC(=O)OCCOC'))
    print(output)
    print(type(output))
    print(output.shape)
