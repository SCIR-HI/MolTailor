import os
import sys
from pathlib import Path, PosixPath
from typing import Union, Any, Tuple, Dict, List, Callable, Sequence

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

import torch
import optuna

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.core.mixins import HyperparametersMixin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Timer, LearningRateMonitor, BatchSizeFinder, LearningRateFinder
from optuna.integration import PyTorchLightningPruningCallback

from data_modules import MultiTaskPreTrainingDataModule
from models.load import load_model, load_tokenizer
from models.multitask import DenForPreTraining
from models.config import DenConfig

ObjectiveFuncType = Callable[[optuna.Trial], Union[float, Sequence[float]]]


class MultiTaskPreTrainingTuner(HyperparametersMixin):
    def __init__(self,
                 model_name_txt,
                 model_name_smi,
                 data_name,
                 seed: int = 42,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 lr: float = 5e-5,
                 warmup_proportion: float = 0.1,
                 max_epochs: int = 10,
                 devices: Union[str, int] = 2,
                 precision: str = '32-true',
                 root_dir: str = './workspace',
                 batch_size_find: bool = True,
                 lr_find: bool = False,
                 lr_max: float = 1e-4,
                 lr_min: float = 1e-6,
                 n_trials: int = 10,
                 limit_train_batches: float = 0.01,
                 limit_val_batches: float = 0.1) -> None:
        super().__init__()

        self.save_hyperparameters()
        seed_everything(seed)

        self.txt_tokenizer = load_tokenizer(model_name_txt)
        self.smi_tokenizer = load_tokenizer(model_name_smi)

        self.batch_size = batch_size
        
    def batch_size_find(self):
        dm = MultiTaskPreTrainingDataModule(
            file_name=self.hparams.data_name,
            txt_tokenizer=self.txt_tokenizer,
            smi_tokenizer=self.smi_tokenizer,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

        data_size = int(len(dm.train_dataset) * self.hparams.limit_train_batches)
        num_training_steps = data_size //(self.hparams.batch_size) * self.hparams.max_epochs
        num_warmup_steps = int(num_training_steps * self.hparams.warmup_proportion)
        
        model = DenForPreTraining(
            lr=self.hparams.lr,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )

        trainer = Trainer(
            max_epochs=self.hparams.max_epochs,
            default_root_dir=self.hparams.root_dir,
            logger=False,
            devices=1,
            precision=self.hparams.precision,
            enable_checkpointing=False,
            limit_train_batches=self.hparams.limit_train_batches,
            limit_val_batches=self.hparams.limit_val_batches,
        )
        tuner = Tuner(trainer)

        optimal_batch_size = tuner.scale_batch_size(
            model=model,
            datamodule=dm,
        )

        self.batch_size = optimal_batch_size
        self.hparams.batch_size = optimal_batch_size

        return optimal_batch_size
    
    def construct_objective(self) -> ObjectiveFuncType:
        def objective(trial: optuna.Trial) -> Union[float, Sequence[float]]:
            lr = trial.suggest_float('lr', self.hparams.lr_min, self.hparams.lr_max)

            dm = MultiTaskPreTrainingDataModule(
                file_name=self.hparams.data_name,
                txt_tokenizer=self.txt_tokenizer,
                smi_tokenizer=self.smi_tokenizer,
                batch_size=self.batch_size,
                num_workers=self.hparams.num_workers,
            )

            data_size = int(len(dm.train_dataset) * self.hparams.limit_train_batches)
            num_training_steps = data_size //(self.batch_size * self.hparams.devices) * self.hparams.max_epochs
            num_warmup_steps = int(num_training_steps * self.hparams.warmup_proportion)

            model = DenForPreTraining(
                lr=lr,
                num_training_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
            )

            pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val/loss')

            trainer = Trainer(
                max_epochs=self.hparams.max_epochs,
                default_root_dir=self.hparams.root_dir,
                logger=False,
                devices=self.hparams.devices,
                precision=self.hparams.precision,
                enable_checkpointing=False,
                callbacks=[pruning_callback],
                limit_train_batches=self.hparams.limit_train_batches,
                limit_val_batches=self.hparams.limit_val_batches,
            )

            trainer.fit(model, dm)

            pruning_callback.check_pruned()

            return trainer.callback_metrics['val/loss'].item()
        
        return objective

    def lr_find(self):
        sampler = optuna.samplers.TPESampler(seed=self.hparams.seed)
        storage = optuna.storages.RDBStorage(f'sqlite:///{os.path.join(self.hparams.root_dir, "optuna.db")}')
        study = optuna.create_study(storage=storage, direction='minimize', sampler=sampler)
        objective = self.construct_objective()

        study.optimize(objective, n_trials=self.hparams.n_trials)

        return study.best_params['lr']

    def run(self):
        outputs = dict()
        if self.hparams.batch_size_find:
            optimal_batch_size = self.batch_size_find()
            outputs['optimal_batch_size'] = optimal_batch_size
        if self.hparams.lr_find:
            optimal_lr = self.lr_find()
            outputs['optimal_lr'] = optimal_lr
        return outputs
    

if __name__ == '__main__':
    settings = {
        'model_name_txt': 'PubMedBERT',
        'model_name_smi': 'CHEM-BERT',
        'data_name': 'multi-task-pretrain.pt',
        'seed': 42,
        'batch_size': 2,
        'num_workers': 4,
        'lr': 5e-5,
        'devices': 2,
        'precision': '32-true',
        'max_epochs': 2,
        'warmup_proportion':  0.1,
        'root_dir': 'workspace',
        'limit_train_batches': 0.002,
        'limit_val_batches': 0.002,
        'batch_size_find': True,
        'lr_find': True,
    }

    settings['root_dir'] = str(base_dir / settings['root_dir'])

    tuner = MultiTaskPreTrainingTuner(**settings)
    outputs = tuner.run()

    print(f'optimal_batch_size: {outputs["optimal_batch_size"]}')
    print(f'optimal_lr: {outputs["optimal_lr"]}')