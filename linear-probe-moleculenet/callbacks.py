from typing import Any, Optional
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class CatchKeyboardInterrupt(Callback):
    def __init__(self) -> None:
        super().__init__()
    
    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        exit()

class TestMetricsReset(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(f'Traning RMSE at start of epoch: {pl_module.train_rmse.compute()}')

    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(f'Validation RMSE at start of epoch: {pl_module.val_rmse.compute()}')
