import sys
import torch

from pathlib import Path, PosixPath
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/scibert_scivocab_uncased'

def load_scibert(path: PosixPath = model_path) -> AutoModel:
    config = BertConfig().from_pretrained(path)
    ckpt = torch.load(path / 'pytorch_model.bin')
    
    # remove 'bert.' prefix in ckpt keys
    for k in list(ckpt.keys()):
        if k.startswith('bert.'):
            ckpt[k[len('bert.'):]] = ckpt.pop(k)
            
    # remove ignored keys
    ignored_keys = [
        'bert.pooler.dense.weight',
        'bert.pooler.dense.bias',
        "cls.predictions.bias", 
        "cls.predictions.transform.dense.weight", 
        "cls.predictions.transform.dense.bias", 
        "cls.predictions.transform.LayerNorm.weight", 
        "cls.predictions.transform.LayerNorm.bias", 
        "cls.predictions.decoder.weight", 
        "cls.predictions.decoder.bias",
        "cls.seq_relationship.weight",
        "cls.seq_relationship.bias",
    ]
    
    for k in list(ckpt.keys()):
        if any([k.endswith(x) for x in ignored_keys]):
            ckpt.pop(k)
    model = BertModel(config)
    model.pooler = None
    model.load_state_dict(ckpt)
    
    return model

def load_scibert_tokenizer(path: PosixPath = model_path) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(path)


if __name__ == '__main__':
    model = load_scibert()
    print(model)