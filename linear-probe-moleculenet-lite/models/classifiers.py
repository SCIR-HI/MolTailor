'''
Author: simwit 517992857@qq.com
Date: 2023-07-11 15:52:10
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-07 18:34:39
FilePath: /hqguo/workspace/01-st/finetune-moleculenet/models/classifiers.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
from pathlib import Path
from typing import Dict
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
sys.path.append(str(base_dir.parent))

import torch

from pytorch_lightning import LightningModule

from metrics import CustomAUROC, DeltaAveragePrecision


class LinearClassification(LightningModule):
    def __init__(self, feature_dim: int, task_num: int, batch_size: int = None, lr: float = 1e-5) -> None:
        super().__init__()
        
        self.classifier = torch.nn.Linear(feature_dim, task_num)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.activation = torch.nn.Sigmoid()

        self.train_roc_auc = CustomAUROC(task_num)
        self.val_roc_auc = CustomAUROC(task_num)
        self.test_roc_auc = CustomAUROC(task_num)

        self.train_dap = DeltaAveragePrecision(task_num)
        self.val_dap = DeltaAveragePrecision(task_num)
        self.test_dap = DeltaAveragePrecision(task_num)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        return self.classifier(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> torch.Tensor:
        x, y, mask = batch['feature'], batch['label'], batch['label_mask']
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        loss = (loss * mask).sum() / mask.sum()
        
        y_hat = self.activation(y_hat)
        self.train_roc_auc.update(y_hat, y, mask)
        self.train_dap.update(y_hat, y, mask)
        
        self.log('train/loss', loss)
        self.log('train/roc_auc', self.train_roc_auc, on_step=False, on_epoch=True)
        self.log('train/dap', self.train_dap, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> None:
        x, y, mask = batch['feature'], batch['label'], batch['label_mask']
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        loss = (loss * mask).sum() / mask.sum()
        
        y_hat = self.activation(y_hat)
        self.val_roc_auc.update(y_hat, y, mask)
        self.val_dap.update(y_hat, y, mask)
        
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/roc_auc', self.val_roc_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/dap', self.val_dap, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> None:
        x, y, mask = batch['feature'], batch['label'], batch['label_mask']
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        loss = (loss * mask).sum() / mask.sum()
        
        y_hat = self.activation(y_hat)
        self.test_roc_auc.update(y_hat, y, mask)
        self.test_dap.update(y_hat, y, mask)
        
        self.log('test/loss', loss)
        self.log('test/roc_auc', self.test_roc_auc, on_step=False, on_epoch=True)
        self.log('test/dap', self.test_dap, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        print('Training with lr:', self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class LinearReassification(LightningModule):
    def __init__(self, feature_dim: int, task_num: int, batch_size: int = None, lr: float = 1e-5) -> None:
        super().__init__()
        
        self.classifier = torch.nn.Linear(feature_dim, task_num)
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.activation = torch.nn.Sigmoid()

        self.train_roc_auc = CustomAUROC(task_num)
        self.val_roc_auc = CustomAUROC(task_num)
        self.test_roc_auc = CustomAUROC(task_num)

        self.train_dap = DeltaAveragePrecision(task_num)
        self.val_dap = DeltaAveragePrecision(task_num)
        self.test_dap = DeltaAveragePrecision(task_num)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        return self.classifier(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> torch.Tensor:
        x, y, mask = batch['feature'], batch['label'], batch['label_mask']
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        loss = (loss * mask).sum() / mask.sum()

        # from 0~1 to -0.5~0.5
        y_hat = (y_hat - 0.5) * 2
        y_hat = self.activation(y_hat)
        self.train_roc_auc.update(y_hat, y, mask)
        self.train_dap.update(y_hat, y, mask)
        
        self.log('train/loss', loss)
        self.log('train/roc_auc', self.train_roc_auc, on_step=False, on_epoch=True)
        self.log('train/dap', self.train_dap, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> None:
        x, y, mask = batch['feature'], batch['label'], batch['label_mask']
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        loss = (loss * mask).sum() / mask.sum()

        y_hat = (y_hat - 0.5) * 2 
        y_hat = self.activation(y_hat)
        self.val_roc_auc.update(y_hat, y, mask)
        self.val_dap.update(y_hat, y, mask)
        
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/roc_auc', self.val_roc_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/dap', self.val_dap, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> None:
        x, y, mask = batch['feature'], batch['label'], batch['label_mask']
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        loss = (loss * mask).sum() / mask.sum()
        
        y_hat = (y_hat - 0.5) * 2
        y_hat = self.activation(y_hat)
        self.test_roc_auc.update(y_hat, y, mask)
        self.test_dap.update(y_hat, y, mask)
        
        self.log('test/loss', loss)
        self.log('test/roc_auc', self.test_roc_auc, on_step=False, on_epoch=True)
        self.log('test/dap', self.test_dap, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        print('Training with lr:', self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


