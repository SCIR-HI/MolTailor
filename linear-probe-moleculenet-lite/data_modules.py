from typing import Any, Tuple

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class GeneralDataModule(LightningDataModule):
    def __init__(self, 
                 dataset: Tuple[Dataset, Dataset, Dataset],
                 train_batch_size: int = 32,
                 inference_batch_size: int = 64,
                 num_workers: int = 4) -> None:
        super().__init__()
        self.train_dataset, self.val_dataset, self.test_dataset = dataset
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
    

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, 
                          batch_size=self.train_batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.inference_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.inference_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
