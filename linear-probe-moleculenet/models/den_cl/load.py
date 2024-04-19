'''
Author: simwit 517992857@qq.com
Date: 2023-08-03 10:57:34
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-04 10:16:22
FilePath: /workspace/01-st/finetune-moleculenet/models/den_cl/load.py
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
model_path = base_path / 'models/DEN-CL'
txt_encoder_path = base_path / 'models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
encoder_type = 'txt'


def load_den_cl(path: PosixPath = model_path, encoder_type: str = encoder_type):
    ckpt_id = 'm34sj8tr'
    ckpt_all = torch.load(path / ckpt_id / 'last.ckpt', map_location='cpu')['state_dict']
    if encoder_type == 'smi':
        ckpt = {}
        for key, value in ckpt_all.items():
            # prefix = 'model.smi_encoder.'
            if key.startswith('model.smi_encoder.'):
                key = key[len('model.smi_encoder.'):]
                ckpt[key] = value
        
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        model = ChemBert(**config)

        # for name, param in model.named_parameters():
        #     print(f'{name}: {param.shape}')
        # torch.save(ckpt, model_path / ckpt_id / 'smi_encoder.ckpt')
        model.load_state_dict(ckpt)

    elif encoder_type == 'txt':
        ckpt = {}
        for key, value in ckpt_all.items():
            # prefix = 'model.smi_encoder.'
            if key.startswith('model.txt_encoder.'):
                key = key[len('model.txt_encoder.'):]
                ckpt[key] = value 
        
        
        model = BertModel(BertConfig.from_pretrained(txt_encoder_path))
        model.pooler = None
        # torch.save(ckpt, model_path / ckpt_id / 'txt_encoder.ckpt')
        model.load_state_dict(ckpt)

    return model

def load_den_cl_tokenizer(path: PosixPath = model_path, encoder_type: str = encoder_type):
    if encoder_type == 'smi':
        return ChemBertTokenizer(vocab_path=path / 'vocab.json')
    elif encoder_type == 'txt':
        return BertTokenizer.from_pretrained(txt_encoder_path)
    

if __name__ == '__main__':
    encoder_type = 'smi'
    # encoder_type = 'txt'

    model = load_den_cl(encoder_type=encoder_type)

            