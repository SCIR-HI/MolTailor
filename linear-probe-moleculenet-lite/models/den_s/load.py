'''
Author: simwit 517992857@qq.com
Date: 2023-07-29 10:37:04
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-07-29 15:53:05
FilePath: /hqguo/workspace/01-st/finetune-moleculenet/models/den_s/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''


import sys
import torch

from pathlib import Path, PosixPath
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/DEN-S'

def load_den_s(path: PosixPath = model_path) -> AutoModel:
    config = BertConfig()
    # ckpt_id = 'svsmsg6p'
    # ckpt_id = '675hzfcj'
    # ckpt_id = '7sx8zplf'
    # ckpt_id = 'hou4yhkl'
    ckpt_id = 'cuqi8e2u'
    ckpt = torch.load(path / ckpt_id / 'last.ckpt')['state_dict']
    
    # remove 'model.bert.' prefix in ckpt keys
    for k in list(ckpt.keys()):
        if k.startswith('model.bert.'):
            ckpt[k[len('model.bert.'):]] = ckpt.pop(k)
    
    # remove 'model.' prefix in ckpt keys
    for k in list(ckpt.keys()):
        if k.startswith('model.'):
            ckpt[k[len('model.'):]] = ckpt.pop(k)
            
    # remove ignored keys
    ignored_keys = [
        "cls.predictions.bias", 
        "cls.predictions.transform.dense.weight", 
        "cls.predictions.transform.dense.bias", 
        "cls.predictions.transform.LayerNorm.weight", 
        "cls.predictions.transform.LayerNorm.bias", 
        "cls.predictions.decoder.weight", 
        "cls.predictions.decoder.bias",
    ]
    
    for k in list(ckpt.keys()):
        if any([k.endswith(x) for x in ignored_keys]):
            ckpt.pop(k)
    model = BertModel(config)
    model.pooler = None
    model.load_state_dict(ckpt)
    
    return model

def load_den_s_tokenizer(path: PosixPath = model_path) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(path)


if __name__ == '__main__':
    model = load_den_s()
    tokenizer = load_den_s_tokenizer()
    print(model)
    print(tokenizer)