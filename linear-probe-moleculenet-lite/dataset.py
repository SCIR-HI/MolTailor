'''
Author: simwit 517992857@qq.com
Date: 2023-07-21 17:03:59
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-07-25 16:55:34
FilePath: /workspace/01-st/finetune-moleculenet/dataset.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
'''
Author: simwit 517992857@qq.com
Date: 2023-07-03 20:57:30
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-07-21 17:01:32
FilePath: /workspace/01-st/finetune-moleculenet/data/utils/construct.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
from pathlib import Path
from typing import Any, Tuple

import torch

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import Dataset

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
data_dir = base_dir / 'data'


from split import split_data


class MoleculeNetFeatureDataset(Dataset):
    def __init__(self, task_type: str, feature: np.ndarray, label: pd.DataFrame = None) -> None:
        super().__init__()
        self.task_type = task_type
        self.feature = feature
        self.label = label
        self.feature_dim = feature.shape[1]
        self.task_num = label.shape[1] if label is not None else None
    
    def __getitem__(self, index) -> Any:
        data = {'feature': torch.tensor(self.feature[index])}
        if self.label is not None:
            label = self.label.iloc[index].to_list()
            data['label'] = torch.tensor([0 if np.isnan(x) else x for x in label], dtype=torch.float)
            if self.task_type == 'classification':
                data['label_mask'] = torch.tensor([not np.isnan(x) for x in label], dtype=torch.bool)
            
        return data

    def __len__(self) -> int:
        return self.feature.shape[0]
    



def construct_dataset(task_name: str, 
                        task_type: str,
                        feature_name: str, 
                        split_type: str='scaffold', 
                        balanced: bool=True,
                        sizes: Tuple[float, float, float] = [0.8, 0.1, 0.1],
                        seed: int = 0) -> Tuple[Dataset, Dataset, Dataset]:

    pl.seed_everything(seed)
    
    data_path = data_dir / (task_name + '.csv')
    feature_path = data_dir / 'feature' / (task_name + '-' + feature_name + '.npy')
    print(f'loading data from {data_path}')
    print(f'loading feature from {feature_path}')
    
    df = pd.read_csv(data_path)
    feature = np.load(feature_path)

    assert df.shape[0] == feature.shape[0], 'feature and label should have the same length'
    
    train_idx, val_idx, test_idx = split_data(df['smiles'].to_list(), 
                                                split_type=split_type, 
                                                balanced=balanced,
                                                sizes=sizes,)
    
    assert len(train_idx) + len(val_idx) + len(test_idx) == df.shape[0], 'split error'
    
    train_dataset = MoleculeNetFeatureDataset(task_type, feature[train_idx], df.loc[train_idx, df.columns.to_list()[1:]])
    val_dataset = MoleculeNetFeatureDataset(task_type, feature[val_idx], df.loc[val_idx, df.columns.to_list()[1:]])
    test_dataset = MoleculeNetFeatureDataset(task_type, feature[test_idx], df.loc[test_idx, df.columns.to_list()[1:]])
    
    return train_dataset, val_dataset, test_dataset


def test_construct_dataset(train_dataset, val_dataset, test_dataset):
    print(f'len of train_dataset: {len(train_dataset)}')
    print(f'len of val_dataset: {len(val_dataset)}')
    print(f'len of test_dataset: {len(test_dataset)}')
    
    # train_dataset
    random_index = np.random.randint(0, len(train_dataset))
    print(f'{random_index}th sample of train_dataset: {train_dataset[random_index]}')
    
    # val_dataset
    random_index = np.random.randint(0, len(val_dataset))
    print(f'{random_index}th sample of val_dataset: {val_dataset[random_index]}')
    
    # test_dataset
    random_index = np.random.randint(0, len(test_dataset))
    print(f'{random_index}th sample of test_dataset: {test_dataset[random_index]}')
    
    print(f'feature shape: {train_dataset[random_index]["feature"].shape}')
    print(f'label shape: {train_dataset[random_index]["label"].shape}')


if __name__ == '__main__':
    test_construct_dataset(*construct_dataset(
        task_name='bbbp',
        task_type='classification',
        # feature_name='ChemBERTa-77M-MTR',
        feature_name='Random',
        split_type='scaffold',
    ))
