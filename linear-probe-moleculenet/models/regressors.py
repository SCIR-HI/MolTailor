'''
Author: simwit 517992857@qq.com
Date: 2023-07-24 21:32:05
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-12 17:40:38
FilePath: /workspace/01-st/finetune-moleculenet/models/regressors.py
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
from torchmetrics import MeanSquaredError, MeanAbsoluteError


class LinearRegression(LightningModule):
    def __init__(self, feature_dim: int, task_num: int, batch_size: int = None, lr: float = 1e-5) -> None:
        super().__init__()
        
        self.regressor = torch.nn.Linear(feature_dim, task_num)
        self.criterion = torch.nn.MSELoss()

        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        return self.regressor(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> torch.Tensor:
        x, y = batch['feature'], batch['label']
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        
        self.train_rmse.update(y_hat, y)
                
        self.log('train/loss', loss)
        self.log('train/rmse', self.train_rmse)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> None:
        x, y = batch['feature'], batch['label']
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)

        self.val_rmse.update(y_hat, y)

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/rmse', self.val_rmse, prog_bar=True)


    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> None:
        x, y = batch['feature'], batch['label']
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        
        self.test_rmse.update(y_hat, y)
        
        self.log('test/loss', loss)
        self.log('test/rmse', self.test_rmse)
        
    
    def configure_optimizers(self):
        print('Training with lr:', self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MultiLinearRegression(LightningModule):
    def __init__(self, feature_dim: int, task_num: int, batch_size: int = None, lr: float = 1e-5, scaler = None) -> None:
        super().__init__()
        
        self.scaler = scaler
        self.regressor = torch.nn.Linear(feature_dim, task_num)
        self.criterion = torch.nn.L1Loss()

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        return self.regressor(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> torch.Tensor:
        x, y = batch['feature'], batch['label']
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)

        y_hat = torch.tensor(self.scaler.inverse_transform(y_hat.detach().cpu().numpy())).to(y.device)
        self.train_mae.update(y_hat, y)
                
        self.log('train/loss', loss)
        self.log('train/rmse', self.train_mae)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> None:
        x, y = batch['feature'], batch['label']
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)

        y_hat = torch.tensor(self.scaler.inverse_transform(y_hat.detach().cpu().numpy())).to(y.device)
        self.val_mae.update(y_hat, y)

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/rmse', self.val_mae, prog_bar=True)


    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> None:
        x, y = batch['feature'], batch['label']
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        
        y_hat = torch.tensor(self.scaler.inverse_transform(y_hat.detach().cpu().numpy())).to(y.device)
        self.test_mae.update(y_hat, y)
        
        self.log('test/loss', loss)
        self.log('test/rmse', self.test_mae)
        
    
    def configure_optimizers(self):
        print('Training with lr:', self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer