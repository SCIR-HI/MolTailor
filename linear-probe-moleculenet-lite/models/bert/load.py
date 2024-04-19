'''
Author: simwit 517992857@qq.com
Date: 2023-08-03 16:57:04
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-03 17:07:01
FilePath: /workspace/01-st/finetune-moleculenet/models/bert/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
import torch

from pathlib import Path, PosixPath
from transformers import BertTokenizer, BertModel, BertConfig, BertModel


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/bert-base-uncased'

def load_bert(path: PosixPath = model_path) -> BertModel:
    model = BertModel.from_pretrained(path)
    model.pooler = None
    # for name, param in model.named_parameters():
    #     print(f'{name}, {param.shape}')

    return model


def load_bert_tokenizer(path: PosixPath = model_path) -> BertTokenizer:
    return BertTokenizer.from_pretrained(path)

