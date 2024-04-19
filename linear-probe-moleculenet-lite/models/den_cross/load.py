'''
Author: simwit 517992857@qq.com
Date: 2023-08-03 10:57:34
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-04 18:52:14
FilePath: /workspace/01-st/finetune-moleculenet/models/den_cross/load.py
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

from transformers import BertTokenizer, BertModel, BertConfig


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/DEN-Cross'
txt_encoder_path = base_path / 'models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
encoder_type = 'smi'
# encoder_type = 'txt'


def load_den_cross(path: PosixPath = model_path, encoder_type: str = encoder_type) -> ChemBert:
    # ckpt_id = 'cwc8lwr2'
    # ckpt_id = 'vwyr9iii'
    ckpt_id = 'z4d8lg2a-best'
    ckpt_all = torch.load(path / ckpt_id / 'last.ckpt', map_location='cpu')['state_dict']
    # for key, value in ckpt_all.items():
    #     print(f'{key}: {value.shape}')

    if encoder_type == 'smi':
        ckpt = {}
        for key, value in ckpt_all.items():
            # prefix = 'model.smi_encoder.'
            if key.startswith('model.den.smi_encoder.'):
                key = key[len('model.den.smi_encoder.'):]
                ckpt[key] = value
        # for key, value in ckpt.items():
        #     print(f'{key}: {value.shape}')
        
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        model = ChemBert(**config)

        # for name, param in model.named_parameters():
        #     print(f'{name}: {param.shape}')
        model.load_state_dict(ckpt)
    
    elif encoder_type == 'txt':
        ckpt = {}
        for key, value in ckpt_all.items():
            if key.startswith('model.den.txt_encoder.'):
                key = key[len('model.den.txt_encoder.'):]
                ckpt[key] = value
        
        model = BertModel(BertConfig.from_pretrained(txt_encoder_path))
        model.pooler = None

        model.load_state_dict(ckpt)

    return model

def load_den_cross_tokenizer(path: PosixPath = model_path, encoder_type: str = encoder_type) -> ChemBertTokenizer:
    if encoder_type == 'smi':
        return ChemBertTokenizer(vocab_path=path / 'vocab.json')
    elif encoder_type == 'txt':
        return BertTokenizer.from_pretrained(txt_encoder_path)
            