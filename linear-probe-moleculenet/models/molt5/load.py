'''
Author: simwit 517992857@qq.com
Date: 2023-08-07 21:49:43
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-07 22:06:27
FilePath: /workspace/01-st/finetune-moleculenet/models/molt5/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
import torch

from pathlib import Path, PosixPath
from transformers import AutoTokenizer, T5ForConditionalGeneration


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/molt5-base'


def load_molt5(path: PosixPath = model_path) -> T5ForConditionalGeneration:
    model = T5ForConditionalGeneration.from_pretrained(path)
    return model

def load_molt5_tokenizer(path: PosixPath = model_path) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path, model_max_length=512)
    return tokenizer


if __name__ == '__main__':
    smiles = 'CCC(O)C(=O)O'
    tokenizer = load_molt5_tokenizer()
    model = load_molt5()

    inputs = tokenizer(smiles, return_tensors='pt')
    encoder_outputs = model.encoder(**inputs)

    encoder_outputs.last_hidden_state.mean(dim=1).shape

