'''
Author: simwit 517992857@qq.com
Date: 2023-07-21 22:19:35
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-07-21 22:21:02
FilePath: /workspace/01-st/finetune-moleculenet/models/chemberta_77m_mtr/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
import torch

from pathlib import Path, PosixPath
from transformers import AutoTokenizer, AutoModel, RobertaModel, RobertaConfig, RobertaForSequenceClassification


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/ChemBERTa-77M-MTR'

def load_chemberta_77m_mtr(path: PosixPath = model_path) -> AutoModel:
    config = RobertaConfig().from_pretrained(path)
    ckpt = torch.load(path / 'pytorch_model.bin')
    
    # remove 'roberta.' prefix in ckpt keys
    for k in list(ckpt.keys()):
        if k.startswith('roberta.'):
            ckpt[k[len('roberta.'):]] = ckpt.pop(k)
            
    # remove ignored keys
    ignored_keys = [
        "embeddings.position_ids", 
        "norm_mean", 
        "norm_std", 
        "regression.dense.weight", 
        "regression.dense.bias", 
        "regression.out_proj.weight", 
        "regression.out_proj.bias"
    ]
    
    for k in list(ckpt.keys()):
        if any([k.endswith(x) for x in ignored_keys]):
            ckpt.pop(k)
    model = RobertaModel(config)
    model.pooler = None
    model.load_state_dict(ckpt)
    
    return model

def load_chemberta_77m_mtr_tokenizer(path: PosixPath = model_path) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(path)