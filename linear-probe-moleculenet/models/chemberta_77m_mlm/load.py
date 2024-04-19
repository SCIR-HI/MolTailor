'''
Author: simwit 517992857@qq.com
Date: 2023-07-27 09:49:14
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-07-27 09:51:46
FilePath: /workspace/01-st/finetune-moleculenet/models/chemberta_77m_mlm/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''

import sys
import torch

from pathlib import Path, PosixPath
from transformers import AutoTokenizer, AutoModel, RobertaConfig, RobertaModel


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/ChemBERTa-77M-MLM'

def load_chemberta_77m_mlm(path: PosixPath = model_path) -> AutoModel:
    config = RobertaConfig().from_pretrained(path)
    ckpt = torch.load(path / 'pytorch_model.bin')
    
    # remove 'bert.' prefix in ckpt keys
    for k in list(ckpt.keys()):
        if k.startswith('roberta.'):
            ckpt[k[len('roberta.'):]] = ckpt.pop(k)
            
    # remove ignored keys
    ignored_keys = [
        "lm_head.bias", 
        "lm_head.dense.weight", 
        "lm_head.dense.bias", 
        "lm_head.layer_norm.weight", 
        "lm_head.layer_norm.bias", 
        "lm_head.decoder.weight", 
        "lm_head.decoder.bias", 
        "embeddings.position_ids",
    ]
    
    for k in list(ckpt.keys()):
        if any([k.endswith(x) for x in ignored_keys]):
            ckpt.pop(k)
    model = RobertaModel(config)
    model.pooler = None
    model.load_state_dict(ckpt)
    
    return model

def load_chemberta_77m_mlm_tokenizer(path: PosixPath = model_path) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(path)


if __name__ == '__main__':
    model = load_chemberta_77m_mlm()
    print(model)