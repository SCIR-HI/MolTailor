'''
Author: simwit 517992857@qq.com
Date: 2023-07-21 16:59:05
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-12 18:05:50
FilePath: /workspace/01-st/finetune-moleculenet/train.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import os
import sys
import time
from pathlib import Path
from typing import Union, List

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir.parent))

import torch
import pytorch_lightning as pl
import pandas as pd

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Timer
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.core.mixins import HyperparametersMixin

from callbacks import TestMetricsReset
from dataset import construct_dataset
from data_modules import GeneralDataModule
from models.classifiers import LinearClassification, LinearReassification
from models.regressors import LinearRegression, MultiLinearRegression
from tune import MoleculeNetTuner
from sklearn.preprocessing import StandardScaler


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


class MoleculeNetTrainer(HyperparametersMixin):
    def __init__(self, 
                feature_name: str,
                task_name: str,
                seed: int = 0,
                lr: float = 0.001,
                train_batch_size: int = 32,
                inference_batch_size: int = 64,
                num_workers: int = 4,
                max_epochs: int = 10, 
                logger_name: str='', 
                logger_offline: bool = True,
                root_dir: str = './workspace',       
                devices: Union[int, str, List[int]] = 'auto',
                use_reassification: bool = False) -> None:
        
        super().__init__()
        self.task_type = name2type[task_name]
        self.split_type = name2split[task_name]
        self.save_hyperparameters()
        self.name = f'[{feature_name}]-{task_name}-{self.split_type}-{seed}' if not logger_name else logger_name

        datasets = construct_dataset(
            task_name=task_name,
            task_type=self.task_type,
            feature_name=feature_name, 
            split_type=self.split_type, 
            seed=seed
        )
        feature_dim = datasets[0].feature_dim
        task_num = datasets[0].task_num
        
        self.scaler = None
        if task_name in ['qm7', 'qm8', 'qm9']:
            self.scaler = StandardScaler()
            self.scaler.fit(datasets[0].label)
            datasets[0].label = pd.DataFrame(self.scaler.transform(datasets[0].label))

        pl.seed_everything(seed)
        self.dm = GeneralDataModule(
            datasets, 
            train_batch_size=train_batch_size,
            inference_batch_size=inference_batch_size, 
            num_workers=num_workers
        )
        if self.task_type == 'regression':
            if task_name in ['qm7', 'qm8', 'qm9']:
                self.model = MultiLinearRegression(
                    feature_dim=feature_dim, 
                    task_num=task_num,
                    lr = lr,
                    scaler=self.scaler,
                )
            else:
                self.model = LinearRegression(
                    feature_dim=feature_dim, 
                    task_num=task_num,
                    lr = lr,
                )
        elif self.task_type == 'classification':
            if use_reassification:
                self.model = LinearReassification(
                    feature_dim=feature_dim, 
                    task_num=task_num,
                    lr = lr
                )
            else:
                self.model = LinearClassification(
                    feature_dim=feature_dim, 
                    task_num=task_num,
                    lr = lr
                )
        else:
            print(f'Error: task_type {self.task_type} is not supported!')
            exit(1)



    def train(self) -> None:
        # save_dir = f"{self.hparams.root_dir}/log/{self.name}_{time.time()}"
        wandb_logger = WandbLogger(
            project="01-st-finetune-moleculenet", 
            name=self.name, 
            save_dir=self.hparams.root_dir,
            offline=self.hparams.logger_offline,
        )

        if self.task_type == 'classification':
            model_checkpoint = ModelCheckpoint(
                dirpath=os.path.join(self.hparams.root_dir, f'checkpoints/{wandb_logger.experiment.id}'),
                filename='epoch:{epoch:02d}-step:{step:04d}-roc_auc:{val/roc_auc:.4f}',
                monitor='val/roc_auc',
                mode='max',
                auto_insert_metric_name=False,
            )
            early_stopping = EarlyStopping(monitor='val/roc_auc', patience=3, mode='max')
        elif self.task_type == 'regression':
            model_checkpoint = ModelCheckpoint(
                dirpath=os.path.join(self.hparams.root_dir, f'checkpoints/{wandb_logger.experiment.id}'),
                filename='epoch:{epoch:02d}-step:{step:04d}-rmse:{val/rmse:.4f}',
                monitor='val/rmse',
                mode='min',
                auto_insert_metric_name=False,
            )
            early_stopping = EarlyStopping(monitor='val/rmse', patience=3, mode='min') 

        timer = Timer()

        trainer = pl.Trainer(
            max_epochs=self.hparams.max_epochs, 
            default_root_dir=self.hparams.root_dir, 
            callbacks=[
                model_checkpoint,
                early_stopping,
                timer,
            ], 
            logger=wandb_logger,
            devices=self.hparams.devices,
            enable_progress_bar=False,
        )
        trainer.fit(
            model= self.model, 
            datamodule= self.dm,
        )
        trainer.test(datamodule= self.dm, ckpt_path='best')
        
        # 关闭wandb_logger
        wandb_logger.experiment.finish()

        # 返回test/roc_auc
        if self.task_type == 'classification':
            return wandb_logger.experiment.id, trainer.callback_metrics['test/roc_auc'], trainer.callback_metrics['test/dap'], timer.time_elapsed('train')
        elif self.task_type == 'regression':
            return wandb_logger.experiment.id, trainer.callback_metrics['test/rmse'], timer.time_elapsed('train')

    def run(self) -> tuple:
        return self.train()




if __name__ == '__main__':
    # test    
    settings = {
        'feature_name': 'ChemBERTa-10M-MTR',
        'task_name': 'esol',
        'seed': 1238,
        'train_batch_size': 64,
        'inference_batch_size': 128,
        'num_workers': 4,
        'max_epochs': 20, 
        'root_dir': './workspace',
        'devices': 'auto',
        
        'lr':  0.001,
        'logger_name': 'test', 
    }

    # settings['task_name'] = 'esol'
    # settings['feature_name'] = 'Random'
    trainer = MoleculeNetTrainer(**settings)
    roc_auc = trainer.run()
    print(f'roc_auc: {roc_auc}')






