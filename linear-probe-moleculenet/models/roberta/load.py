'''
Author: simwit 517992857@qq.com
Date: 2023-08-03 17:08:03
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-03 17:19:52
FilePath: /workspace/01-st/finetune-moleculenet/models/roberta/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''

import sys
import torch

from pathlib import Path, PosixPath
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/roberta-base'

def load_roberta(path: PosixPath = model_path) -> RobertaModel:
    model = RobertaModel.from_pretrained(path)
    model.pooler = None
    # for name, param in model.named_parameters():
    #     print(f'{name}, {param.shape}')

    return model


def load_roberta_tokenizer(path: PosixPath = model_path) -> RobertaTokenizer:
    return RobertaTokenizer.from_pretrained(path)
