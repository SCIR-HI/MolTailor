'''
Author: simwit 517992857@qq.com
Date: 2023-07-24 10:38:46
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-07-25 17:27:27
FilePath: /workspace/01-st/finetune-moleculenet/multi_seeds.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''

import os
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir.parent))

from typing import Union, List, Tuple, Dict
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from train import MoleculeNetTrainer
from tune import MoleculeNetTuner

name2type = {
    'bbbp': 'classification',
    'clintox': 'classification',
    'hiv': 'classification',
    'tox21': 'classification',
    'esol': 'regression',
    'freesolv': 'regression',
    'lipophilicity': 'regression',
    'qm7': 'regression',
    'qm8': 'regression',
    'qm9': 'regression'
}

name2split = {
    'bbbp': 'scaffold',
    'clintox': 'random',
    'hiv': 'scaffold',
    'tox21': 'random',
    'esol': 'random',
    'freesolv': 'random',
    'lipophilicity': 'random',
    'qm7': 'random',
    'qm8': 'random',
    'qm9': 'random'
}

class MultiSeeedsFinetune:
    def __init__(self,
               feature_name: str,
               task_name: str,
               train_batch_size: int = 32,
               inference_batch_size: int = 64,
               num_workers: int = 4,
               max_epochs: int = 10,
               root_dir: str = './workspace',
               devices: Union[int, str, List[int]] = 'auto',
               lr: float = 0.001,
               logger_name: str = '',
               logger_offline: bool = True,
               use_tune: bool = True,
               lr_max: float = 1e-2,
               lr_min: float = 1e-5,
               n_trials: int = 2,
               seeds: List[int] = [0],
               use_reassification: bool = False) -> None:
        self.feature_name = feature_name
        self.task_name = task_name
        self.task_type = name2type[task_name]
        self.split_type = name2split[task_name]
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.root_dir = root_dir
        self.devices = devices
        self.lr = lr
        self.logger_name = logger_name
        self.logger_offline = logger_offline
        self.use_tune = use_tune
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.n_trials = n_trials
        self.seeds = seeds
        self.use_reassification = use_reassification

        self.settings_train = {
            'feature_name': self.feature_name,
            'task_name': self.task_name,
            'train_batch_size': self.train_batch_size,
            'inference_batch_size': self.inference_batch_size,
            'num_workers': self.num_workers,
            'max_epochs': self.max_epochs,
            'root_dir': self.root_dir,
            'devices': self.devices,
            
            'lr': self.lr,
            'logger_name': self.logger_name,
            'logger_offline': self.logger_offline,
            'use_reassification': self.use_reassification,
        }

        if self.use_tune:
            self.settings_tune = {
                'feature_name': self.feature_name,
                'task_name': self.task_name,
                'train_batch_size': self.train_batch_size,
                'inference_batch_size': self.inference_batch_size,
                'num_workers': self.num_workers,
                'max_epochs': self.max_epochs,
                'root_dir': self.root_dir,
                'devices': self.devices,

                'lr_max': self.lr_max,
                'lr_min': self.lr_min,
                'n_trials': self.n_trials,
                'use_reassification': self.use_reassification,
            }

    def iter_run(self, seed: int) -> Tuple[str, float, float, float]:
        if self.use_tune:
            self.settings_tune['seed'] = seed
            tuner = MoleculeNetTuner(**self.settings_tune)
            lr = tuner.run()
            self.settings_train['lr'] = lr

        self.settings_train['seed'] = seed
        trainer = MoleculeNetTrainer(**self.settings_train)

        return trainer.run()
    
    def run(self) -> Dict[str, List]:
        if self.task_type == 'classification':
            results = {
                'run_id': [],
                'roc_auc': [],
                'dap': [],
                'train_time': [],
            }
            for seed in self.seeds:
                run_id, roc_auc, dap, train_time = self.iter_run(seed)
                results['run_id'].append(run_id)
                results['roc_auc'].append(roc_auc)
                results['dap'].append(dap)
                results['train_time'].append(train_time)

        elif self.task_type == 'regression':
            results = {
                'run_id': [],
                'rmse': [],
                'train_time': [],
            }
            for seed in self.seeds:
                run_id, rmse, train_time = self.iter_run(seed)
                results['run_id'].append(run_id)
                results['rmse'].append(rmse)
                results['train_time'].append(train_time)
        
        else:
            raise ValueError(f'Unknown task type: {self.task_type}')
            
        return results
    

if __name__ == '__main__':
    # test
    settings = {
        'feature_name': 'ChemBERTa-77M-MTR',
        'task_name': 'bbbp',
        'train_batch_size': 64,
        'inference_batch_size': 128,
        'num_workers': 4,
        'max_epochs': 20,  
        'root_dir': './workspace',
        'devices': 'auto',
        
        'lr':  0.001,
        'logger_name': '',
        'logger_offline': True,
        
        'use_tune': True,
        'lr_max': 1e-2,
        'lr_min': 1e-5,
        'n_trials': 2,
        'seeds': [1236, 1237, 1238],
        'use_reassification': True,
    }

    # seeds = [1234, 1235, 1236, 1237, 1238]
    # settings['seeds'] = [1234, 1235, 1236, 1237, 1238]
    # settings['feature_name'] = 'Morgan-FP'
    # settings['feature_name'] = 'RDKit-FP'
    settings['feature_name'] = 'MACCS-FP'
    # settings['task_name'] = 'esol'
    start_time = timer()

    multi_seed_finetune = MultiSeeedsFinetune(**settings)
    results = multi_seed_finetune.run()

    end_time = timer()
    print(f'Elapsed time: {(end_time - start_time)/60:.2f} min')

    print(f'[{settings["feature_name"]}]-{settings["task_name"]}')
    print(f'id: {", ".join(results["run_id"])}')
    
    if multi_seed_finetune.task_type == 'classification':
        # roc-auc
        print(f'roc_auc: {np.mean(results["roc_auc"])*100:.2f} ± {np.std(results["roc_auc"])*100:.2f}')
        # dap
        print(f'dap: {np.mean(results["dap"])*100:.2f} ± {np.std(results["dap"])*100:.2f}')
    elif multi_seed_finetune.task_type == 'regression':
        # rmse
        print(f'rmse: {np.mean(results["rmse"]):.4f} ± {np.std(results["rmse"]):.4f}')
    print(f'train_time: {np.mean(results["train_time"]):.2f} ± {np.std(results["train_time"]):.2f}')


