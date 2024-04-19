'''
Author: simwit 517992857@qq.com
Date: 2023-08-04 19:53:47
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-04 20:28:43
FilePath: /hqguo/workspace/01-st/finetune-moleculenet/models/uni_mol/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys

from pathlib import Path, PosixPath


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/Uni-Mol/unimol_tools'
sys.path.append(str(model_path))
sys.path.append(Path(__file__).resolve().parent)

import numpy as np
from unimol_tools import UniMolRepr

def load_uni_mol():
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    return clf


if __name__ == '__main__':
    # single smiles unimol representation
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    smiles = 'c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]'
    smiles_list = [smiles]
    unimol_repr = clf.get_repr(smiles_list)
    # Uni-Mol分子表征,使用cls token
    print(np.array(unimol_repr['cls_repr']).shape)
