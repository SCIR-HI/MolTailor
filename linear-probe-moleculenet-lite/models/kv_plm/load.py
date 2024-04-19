'''
Author: simwit 517992857@qq.com
Date: 2023-08-07 20:43:16
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-07 20:59:28
FilePath: /workspace/01-st/finetune-moleculenet/models/kv_plm/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
import torch

from pathlib import Path, PosixPath
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/KV-PLM'
backbone_path = base_path / 'models/scibert_scivocab_uncased'


from transformers import BertModel, BertTokenizer

def load_kv_plm(path: PosixPath = model_path) -> BertModel:
    ckpt = torch.load(path / 'ckpt_KV.pt')
    # for name, param in ckpt.items():
    #     print(f'{name}: {param.shape}')

    # remove bert. prefix
    for k in list(ckpt.keys()):
        if k.startswith('bert.'):
            ckpt[k[len('bert.'):]] = ckpt.pop(k)

    # remove ignored keys
    ignored_keys = [
        "pooler.dense.weight",
        "pooler.dense.bias",
        "cls.predictions.bias", 
        "cls.predictions.transform.dense.weight", 
        "cls.predictions.transform.dense.bias", 
        "cls.predictions.transform.LayerNorm.weight", 
        "cls.predictions.transform.LayerNorm.bias", 
        "cls.predictions.decoder.weight", 
        "cls.predictions.decoder.bias",
        "cls.seq_relationship.weight",
        "cls.seq_relationship.bias",
        'embeddings.position_ids',
    ]
    for k in list(ckpt.keys()):
        if any([k.endswith(x) for x in ignored_keys]):
            ckpt.pop(k)

    model = BertModel.from_pretrained(backbone_path, add_pooling_layer=False)
    model.load_state_dict(ckpt)

    return model

def load_kv_plm_tokenizer(path: PosixPath = model_path) -> BertTokenizer:
    return BertTokenizer.from_pretrained(backbone_path)