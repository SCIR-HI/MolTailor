'''
Author: simwit 517992857@qq.com
Date: 2023-07-28 21:14:23
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-07-29 16:55:45
FilePath: /workspace/01-st/finetune-moleculenet/models/molformer/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
from pathlib import Path, PosixPath

base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/MoLFormer'

import torch
import pytorch_lightning as pl

# load last.ckpt
def load_molformer(path: PosixPath = model_path) -> pl.LightningModule:
    ckpt = torch.load(path / 'last.ckpt', map_location='cpu')
    config = ckpt['hyper_parameters']
    for name, param in ckpt['state_dict'].items():
        print(f'{name}: {param.shape}')

    ref_ckpt = torch.load('/home/hqguo/workspace/01-st/models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/pytorch_model.bin', map_location='cpu')

    return pl.LightningModule.load_from_checkpoint(path / 'last.ckpt')