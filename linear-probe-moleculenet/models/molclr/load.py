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
model_path = base_path / 'models/MolCLR'

try:
    from ginet import GINet
    from mol2graph import mol2graph
except:
    from .ginet import GINet
    from .mol2graph import mol2graph

import torch


def load_molclr(path: PosixPath = model_path):
    model = GINet(drop_ratio=0.3)
    ckpt = torch.load(path / 'model.pth', map_location='cpu')
    
    ignored_keys = [
        "out_lin.0.weight", 
        "out_lin.0.bias", 
        "out_lin.2.weight", 
        "out_lin.2.bias"
    ]
    for k in ignored_keys:
        ckpt.pop(k)

    model.load_state_dict(ckpt)

    return model


def load_molclr_tokenizer(path: PosixPath = model_path):
    return mol2graph


if __name__ == '__main__':
    model = load_molclr()
    print(model)

    tokenizer = load_molclr_tokenizer()
    tokenized = tokenizer('CCO')
    print(tokenized)

    output = model(tokenized)
    print(output)
    print(output.shape)

     
