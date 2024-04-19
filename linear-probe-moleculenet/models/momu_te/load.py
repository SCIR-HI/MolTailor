'''
Author: simwit 517992857@qq.com
Date: 2023-07-26 15:39:04
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-07-26 16:09:59
FilePath: /workspace/01-st/finetune-moleculenet/models/scibert/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
import torch

from pathlib import Path, PosixPath
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/MoMu'

def load_momu_te(path: PosixPath = model_path) -> AutoModel:
    config = BertConfig().from_pretrained(base_path / 'models/scibert_scivocab_uncased')
    
    ckpt = torch.load(path / 'littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt', map_location='cpu')
    ckpt = ckpt['state_dict']
    
    # remove 'bert.' prefix in ckpt keys
    for k in list(ckpt.keys()):
        if k.startswith('graph_encoder.'):
            ckpt.pop(k)
    # remove replace 'graph_encoder.' with ''
    for k in list(ckpt.keys()):
        if k.startswith('text_encoder.main_model.'):
            ckpt[k.replace('text_encoder.main_model.', '')] = ckpt.pop(k)
            
    # remove ignored keys
    ignored_keys = [
        "graph_proj_head.0.weight", 
        "graph_proj_head.0.bias", 
        "graph_proj_head.2.weight", 
        "graph_proj_head.2.bias", 
        "text_proj_head.0.weight", 
        "text_proj_head.0.bias", 
        "text_proj_head.2.weight",
        "text_proj_head.2.bias",
        "pooler.dense.weight", 
        "pooler.dense.bias", 
        "embeddings.position_ids"
    ]
    
    for k in list(ckpt.keys()):
        if any([k.endswith(x) for x in ignored_keys]):
            ckpt.pop(k)

    model = BertModel(config, add_pooling_layer=False)
    model.load_state_dict(ckpt)
    
    return model

def load_momu_te_tokenizer(path: PosixPath = model_path) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(base_path / 'models/scibert_scivocab_uncased')


if __name__ == '__main__':
    model = load_momu_te()
    print(model)