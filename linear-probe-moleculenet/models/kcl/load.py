'''
Author: simwit 517992857@qq.com
Date: 2023-08-10 19:45:36
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-10 19:45:40
FilePath: /workspace/01-st/finetune-moleculenet/models/kcl/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
'''
Author: simwit 517992857@qq.com
Date: 2023-08-10 17:29:06
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-10 19:23:40
FilePath: /workspace/01-st/finetune-moleculenet/models/kcl/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
import json
from typing import Any
import torch
import pickle

from pathlib import Path, PosixPath


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/KCL'

try:
    from model import KCL
    from smiles2graph import smiles_2_kgdgl
except:
    from .model import KCL
    from .smiles2graph import smiles_2_kgdgl


def load_kcl(path: PosixPath = model_path, 
             encoder_name: str = 'KMPNNGNN_0910_2302_78000th_epoch.pkl',
             readout_name: str = 'Set2Set_0910_2302_78000th_epoch.pkl',
             initial_name: str = 'RotatE_128_64_emb.pkl'):
    with open(model_path / 'config.json') as f:
        config = json.load(f)
    config['model_path'] = path
    config['encoder_name'] = encoder_name
    config['readout_name'] = readout_name
    config['initial_name'] = initial_name
    
    mdoel = KCL(config)

    return mdoel

def load_kcl_tokenizer(path: PosixPath = model_path) -> Any:
    return smiles_2_kgdgl

if __name__ == '__main__':
    model = load_kcl()
    print(model)
    
    
    
    
