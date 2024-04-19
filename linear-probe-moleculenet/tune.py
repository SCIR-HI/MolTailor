#import os
import sys
from pathlib import Path
from typing import Union, List, Callable, Sequence

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir.parent))

import optuna
import pytorch_lightning as pl
import pandas as pd

from optuna.integration import PyTorchLightningPruningCallback

from dataset import construct_dataset
from data_modules import GeneralDataModule
from models.classifiers import LinearClassification, LinearReassification
from models.regressors import LinearRegression, MultiLinearRegression
from sklearn.preprocessing import StandardScaler

ObjectiveFuncType = Callable[[optuna.Trial], Union[float, Sequence[float]]]

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




class MoleculeNetTuner:
    def __init__(self,
                    feature_name: str,
                    task_name: str,
                    seed: int = 0,
                    train_batch_size: int = 32,
                    inference_batch_size: int = 64,
                    num_workers: int = 4,
                    max_epochs: int = 10, 
                    root_dir: str = './workspace',       
                    devices: Union[int, str, List[int]] = 'auto',
                    lr_max: float = 1e-2,
                    lr_min: float = 1e-5,
                    n_trials: int = 10,
                    use_reassification: bool = False) -> None:
        self.feature_name = feature_name
        self.task_name = task_name
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.root_dir = root_dir
        self.devices = devices
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.n_trials = n_trials
        self.use_reassification = use_reassification

        self.task_type = name2type[task_name]
        self.split_type = name2split[task_name]

        self.datasets = construct_dataset(
            task_name=self.task_name,
            task_type=self.task_type,
            feature_name=self.feature_name, 
            split_type=self.split_type, 
            seed=self.seed
        )
        self.feature_dim = self.datasets[0].feature_dim
        self.task_num = self.datasets[0].task_num

        self.scaler = None
        if task_name in ['qm7', 'qm8', 'qm9']:
            self.scaler = StandardScaler()
            self.scaler.fit(self.datasets[0].label)
            self.datasets[0].label = pd.DataFrame(self.scaler.transform(self.datasets[0].label))

    def construct_objective(self) -> ObjectiveFuncType:
        def objective(trial: optuna.trial.Trial) -> float:
            lr = trial.suggest_float('lr', self.lr_min, self.lr_max)

            dm = GeneralDataModule(
                self.datasets, 
                train_batch_size=self.train_batch_size,
                inference_batch_size=self.inference_batch_size, 
                num_workers=self.num_workers
            )
            if self.task_type == 'regression':
                if self.task_name in ['qm7', 'qm8', 'qm9']:
                    model = MultiLinearRegression(
                        feature_dim=self.feature_dim, 
                        task_num=self.task_num,
                        lr = lr,
                        scaler=self.scaler,
                    )
                else:
                    model = LinearRegression(
                        feature_dim=self.feature_dim, 
                        task_num=self.task_num,
                        lr = lr,
                    )
                pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val/rmse')
            elif self.task_type == 'classification':
                if self.use_reassification:
                    model = LinearReassification(
                        feature_dim=self.feature_dim, 
                        task_num=self.task_num,
                        lr = lr
                    )
                else:
                    model = LinearClassification(
                        feature_dim=self.feature_dim, 
                        task_num=self.task_num,
                        lr = lr
                    )
                pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val/roc_auc')
            else:
                print(f'Error: task_type {self.task_type} is not supported!')
                exit(1)
            
            if self.task_name == 'qm9':
                limit_train_batches = 0.1
                limit_val_batches = 0.5
            else:
                limit_train_batches = 1.0
                limit_val_batches = 1.0

            trainer = pl.Trainer(
                max_epochs=self.max_epochs, 
                default_root_dir=self.root_dir, 
                callbacks=[pruning_callback], 
                logger=False,
                devices=self.devices,
                enable_checkpointing=False,
                enable_progress_bar=False,
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches,
            )

            trainer.fit(model, dm)

            if self.task_type == 'regression':
                return trainer.callback_metrics['val/rmse'].item()
            elif self.task_type == 'classification':
                return trainer.callback_metrics['val/roc_auc'].item()

        return objective
    
    def run(self) -> float:
        pl.seed_everything(self.seed)
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        if name2type[self.task_name] == 'regression':
            study = optuna.create_study(direction='minimize', sampler=sampler)
        elif name2type[self.task_name] == 'classification':
            study = optuna.create_study(direction='maximize', sampler=sampler)
        objective = self.construct_objective()

        study.optimize(objective, n_trials=self.n_trials)

        # return best lr
        return study.best_params['lr']





if __name__ == '__main__':
    

    settings = {
        'feature_name': 'ChemBERTa-77M-MTR',
        'task_name': 'bbbp',
        'seed': 1238,
        'train_batch_size': 64,
        'inference_batch_size': 128,
        'num_workers': 4,
        'max_epochs': 20, 
        'root_dir': './workspace',
        'devices': [1],
        
        'lr_max': 1e-2,
        'lr_min': 1e-5,
        'n_trials': 2,
    }

    tuner = MoleculeNetTuner(**settings)
    lr = tuner.run()

    print(f'lr: {lr}')
    # 0.0001492547411656223
