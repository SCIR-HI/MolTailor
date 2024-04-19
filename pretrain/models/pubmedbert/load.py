import sys
import torch

from pathlib import Path, PosixPath
from transformers import AutoConfig, BertForMaskedLM, BertTokenizer, AutoModel


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

def load_pubmedbert(path: PosixPath = model_path) -> AutoModel:
    config = AutoConfig.from_pretrained(path)
    ckpt = torch.load(path / 'pytorch_model.bin')
    
            
    # remove ignored keys
    ignored_keys = [
        "bert.pooler.dense.weight", 
        "bert.pooler.dense.bias",
        "cls.seq_relationship.weight",
        "cls.seq_relationship.bias",
    ]
    
    for k in list(ckpt.keys()):
        if any([k.endswith(x) for x in ignored_keys]):
            ckpt.pop(k)
    model = BertForMaskedLM(config)
    model.load_state_dict(ckpt)
    
    return model

def load_pubmedbert_tokenizer(path: PosixPath = model_path) -> BertTokenizer:
    return BertTokenizer.from_pretrained(path)


if __name__ == '__main__':
    model = load_pubmedbert()
    print(model)

    print('start')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name)
    print('end')
