'''
Author: simwit 517992857@qq.com
Date: 2023-08-11 22:40:54
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-11 22:51:32
FilePath: /workspace/01-st/finetune-moleculenet/models/molclr/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
from pathlib import Path, PosixPath

base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/MoMu'

try:
    from ginet import GINet
    from mol2graph import mol2graph
except:
    from .ginet import GINet
    from .mol2graph import mol2graph

import torch


def load_momu(path: PosixPath = model_path):
    model = GINet()
    ckpt = torch.load(path / 'littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt', map_location='cpu')
    ckpt = ckpt['state_dict']

    # show the keys of ckpt
    # for k, v in ckpt.items():
    #     print(f'{k}: {v.shape}')
    
    # remove keys began with text_encoder.
    for k in list(ckpt.keys()):
        if k.startswith('text_encoder.'):
            ckpt.pop(k)
    # remove replace 'graph_encoder.' with ''
    for k in list(ckpt.keys()):
        if k.startswith('graph_encoder.'):
            ckpt[k.replace('graph_encoder.', '')] = ckpt.pop(k)
    
    ignored_keys = [
        "graph_proj_head.0.weight", 
        "graph_proj_head.0.bias", 
        "graph_proj_head.2.weight", 
        "graph_proj_head.2.bias", 
        "text_proj_head.0.weight", 
        "text_proj_head.0.bias", 
        "text_proj_head.2.weight",
        "text_proj_head.2.bias"
    ]

    for k in ignored_keys:
        ckpt.pop(k)

    model.load_state_dict(ckpt)

    return model


def load_momu_tokenizer(path: PosixPath = model_path):
    return mol2graph


if __name__ == '__main__':
    model = load_momu()
    print(model)

    tokenizer = load_momu_tokenizer()
    tokenized = tokenizer('CCO')
    print(tokenized)

    output = model(tokenized)
    print(output)
    print(output.shape)

     
