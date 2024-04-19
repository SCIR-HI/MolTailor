import sys
import json

import torch

from pathlib import Path, PosixPath

try:
    from .model import ChemBert
    from .tokenizer import ChemBertTokenizer
except ImportError:
    from model import ChemBert
    from tokenizer import ChemBertTokenizer


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/CHEM-BERT'

def load_chembert(path: PosixPath = model_path) -> ChemBert:
    with open(path / 'config.json', 'r') as f:
        config = json.load(f)
    
    ckpt = torch.load(path / 'pretrained_model.pt', map_location='cpu')
    # ckpt = torch.load(base_path / 'models/DEN-CL/m34sj8tr/smi_encoder.ckpt')
    
    model = ChemBert(**config)
    model.load_state_dict(ckpt)
    
    return model

def load_chembert_tokenizer(path: PosixPath = model_path) -> ChemBertTokenizer:
    return ChemBertTokenizer(vocab_path=path / 'vocab.json')


if __name__ == '__main__':
    model = load_chembert()
    tokenizer = load_chembert_tokenizer()
    print(model)
    print(tokenizer)
    print(model.config)