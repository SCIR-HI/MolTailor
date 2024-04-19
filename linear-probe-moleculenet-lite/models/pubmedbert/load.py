'''
Author: simwit 517992857@qq.com
Date: 2023-07-26 15:39:04
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-07-26 15:55:57
FilePath: /workspace/01-st/finetune-moleculenet/models/pubmedbert/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
import torch

from pathlib import Path, PosixPath
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

def load_pubmedbert(path: PosixPath = model_path, ckpt_id: str = '') -> AutoModel:
    config = BertConfig().from_pretrained(path)

    if ckpt_id:
        pass
        ckpt = torch.load(path / 'ckpt' / ckpt_id / 'last.pt')
    else:
        ckpt = torch.load(path / 'pytorch_model.bin')
    
    # remove 'bert.' prefix in ckpt keys
    for k in list(ckpt.keys()):
        if k.startswith('bert.'):
            ckpt[k[len('bert.'):]] = ckpt.pop(k)
        if k.startswith('encoder.'):
            ckpt[k[len('encoder.'):]] = ckpt.pop(k)
            
    # remove ignored keys
    ignored_keys = [
        "cls.predictions.bias", 
        "cls.predictions.transform.dense.weight", 
        "cls.predictions.transform.dense.bias", 
        "cls.predictions.transform.LayerNorm.weight", 
        "cls.predictions.transform.LayerNorm.bias", 
        "cls.predictions.decoder.weight", 
        "cls.predictions.decoder.bias",
        "cls.seq_relationship.weight",
        "cls.seq_relationship.bias",
        "regressor.weight", 
        "regressor.bias"
    ]
    
    for k in list(ckpt.keys()):
        if any([k.endswith(x) for x in ignored_keys]):
            ckpt.pop(k)
    model = BertModel(config)
    model.pooler = None
    model.load_state_dict(ckpt)
    
    return model

def load_pubmedbert_tokenizer(path: PosixPath = model_path) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(path)


if __name__ == '__main__':
    model = load_pubmedbert(ckpt_id='lpbpxt7w')
    print(model)