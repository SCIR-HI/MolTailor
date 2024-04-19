'''
Author: simwit 517992857@qq.com
Date: 2023-07-29 20:35:33
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-07-29 21:49:18
FilePath: /workspace/01-st/finetune-moleculenet/models/chembert/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
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

def load_chembert(path: PosixPath = model_path, ckpt_id: str = '') -> ChemBert:
    with open(path / 'config.json', 'r') as f:
        config = json.load(f)
    if ckpt_id:
        ckpt = torch.load(path / 'ckpt' / ckpt_id / 'last.pt')
        # remove encoder. prefix in keys, don't use replace, it will replace all the encoder. in the key
        for k in list(ckpt.keys()):
            if k.startswith('encoder.'):
                ckpt[k[8:]] = ckpt.pop(k)
        # ignore 
        ignore_keys = ["regressor.weight", "regressor.bias"]
        ckpt = {k: v for k, v in ckpt.items() if k not in ignore_keys}

    else:
        ckpt = torch.load(path / 'pretrained_model.pt')
    
    model = ChemBert(**config)
    model.load_state_dict(ckpt)
    
    return model

def load_chembert_tokenizer(path: PosixPath = model_path) -> ChemBertTokenizer:
    return ChemBertTokenizer(vocab_path=path / 'vocab.json')


if __name__ == '__main__':
    model = load_chembert(ckpt_id='2818iziw')
    # tokenizer = load_chembert_tokenizer()
    # print(model)
    # print(tokenizer)