import os
import sys
from pathlib import Path, PosixPath
from typing import Union, Any, Tuple, Dict, List

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.core.mixins import HyperparametersMixin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Timer, LearningRateMonitor

from data_modules import MultiTaskPreTrainingDataModule
from models.load import load_model, load_tokenizer
from models.multitask import DenForPreTraining
from models.config import DenConfig


class MultiTaskPreTrainingTrainer(HyperparametersMixin):
    def __init__(self,
                 model_name_txt,
                 model_name_smi,
                 data_name,
                 seed: int = 42,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 lr: float = 5e-5,
                 devices: Union[str, int] = 2,
                 precision: str = '32-true',
                 max_epochs: int = 10,
                 warmup_proportion: float = 0.1,
                 root_dir: str = './workspace',
                 overfit_batches: Union[int, float] = 0.0,
                 logger_offline: bool = False) -> None:
        super().__init__()

        self.save_hyperparameters()
        seed_everything(seed)
        self.name = f'[DEN]-{str(seed)}'

        self.txt_tokenizer = load_tokenizer(model_name_txt)
        self.smi_tokenizer = load_tokenizer(model_name_smi)
        
        self.dm = MultiTaskPreTrainingDataModule(
            file_name=data_name,
            txt_tokenizer=self.txt_tokenizer,
            smi_tokenizer=self.smi_tokenizer,
            batch_size=batch_size,
            num_workers=num_workers
        )

        data_size = int(len(self.dm.train_dataset) * (1 if not overfit_batches else overfit_batches))
        num_training_steps = data_size //(batch_size * devices) * max_epochs
        num_warmup_steps = int(num_training_steps * warmup_proportion)

        self.model = DenForPreTraining(
            lr=lr,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )
    
    def train(self):
        wandb_logger = WandbLogger(
            project='01-st-pretrain-den',
            name=self.name,
            save_dir=self.hparams.root_dir,
            offline=self.hparams.logger_offline,
        )

        model_checkpoint = ModelCheckpoint(
            dirpath=os.path.join(self.hparams.root_dir, f'checkpoints/{wandb_logger.experiment.id}'),
            filename='epoch:{epoch}-step:{step}-val_loss:{val/loss:.4f}',
            save_last=True,
            monitor='val/loss',
            mode='min',
            auto_insert_metric_name=False,
        )

        timer = Timer()
        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = Trainer(
            devices=self.hparams.devices,
            precision=self.hparams.precision,
            max_epochs=self.hparams.max_epochs,
            default_root_dir=self.hparams.root_dir,
            callbacks=[model_checkpoint, timer, lr_monitor],
            logger=wandb_logger,
            overfit_batches=self.hparams.overfit_batches,
        )
        trainer.fit(self.model, self.dm)

        wandb_logger.experiment.finish()

        return wandb_logger.experiment.id, trainer.callback_metrics['train/loss'].item(), timer.time_elapsed('train')
   
    def run(self) -> tuple:
        return self.train()
    

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
        'overfit_batches': 0.002, 
        'logger_offline': True,
    }

    settings['root_dir'] = str(base_dir / settings['root_dir'])

    trainer = MultiTaskPreTrainingTrainer(**settings)
    run_id, loss, time = trainer.run()
