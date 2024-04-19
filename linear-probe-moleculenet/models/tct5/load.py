'''
Author: simwit 517992857@qq.com
Date: 2023-08-07 21:49:43
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-08 09:36:43
FilePath: /hqguo/workspace/01-st/finetune-moleculenet/models/t5/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
import torch

from pathlib import Path, PosixPath
from transformers import AutoTokenizer, T5ForConditionalGeneration


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/multitask-text-and-chemistry-t5-base-augm'


def load_tct5(path: PosixPath = model_path) -> T5ForConditionalGeneration:
    model = T5ForConditionalGeneration.from_pretrained(path)
    return model

def load_tct5_tokenizer(path: PosixPath = model_path) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path, model_max_length=512)
    return tokenizer


if __name__ == '__main__':
    smiles = 'CCC(O)C(=O)O'
    tokenizer = load_tct5_tokenizer()
    model = load_tct5()

    inputs = tokenizer(smiles, return_tensors='pt')
    print(inputs)
    print(tokenizer.decode(tokenizer.encode(smiles)))
    encoder_outputs = model.encoder(**inputs)

    print(encoder_outputs.last_hidden_state.mean(dim=1).shape)

    end_token_idx = (inputs['input_ids'] == tokenizer.eos_token_id).nonzero(as_tuple=True)[1]
    # use eos token as feature
    feature = encoder_outputs.last_hidden_state[0, end_token_idx, :].detach().cpu().numpy()
    print(feature.shape)

