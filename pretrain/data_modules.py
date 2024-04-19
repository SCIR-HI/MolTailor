import sys
from pathlib import Path, PosixPath
from typing import Union, Any, Tuple, Dict, List

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
data_dir = base_dir / 'data'

import json

import torch
import numpy as np

from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from dataset import MultiTaskPreTrainingDataset
from data_collator import CustomDataCollatorForMTP
from models.load import load_tokenizer


with open(data_dir / 'task_names.json', 'r') as f:
    task_names = json.load(f)

class MultiTaskPreTrainingDataModule(LightningDataModule):
    def __init__(self, 
                 file_name: str,
                 txt_tokenizer: AutoTokenizer, 
                 smi_tokenizer: AutoTokenizer,
                 sizes: Tuple[float, float] = (0.9, 0.1),
                 batch_size: int = 32,
                 num_workers: int = 4) -> None:
        super().__init__()
        data_path = data_dir / file_name
        self.data = torch.load(data_path)
        # half the data
        # self.data = self.data[:len(self.data) // 2]
        print(f'Load data from {data_path}, data size: {len(self.data)}')
        # reconstruct data
        # build labels-mtr and labels-mtr-mask

        # random split data
        train_size, val_size = sizes
        train_size = int(train_size * len(self.data))
        val_size = len(self.data) - train_size
        train_data, val_data = torch.utils.data.random_split(self.data, [train_size, val_size])

        # normalize labels
        train_labels = torch.stack([ele[2] for ele in train_data])
        train_labels_mask = torch.stack([ele[3] for ele in train_data])
        val_labels = torch.stack([ele[2] for ele in val_data])
        val_labels_mask = torch.stack([ele[3] for ele in val_data])

        scaler_list = []
        
        for i, task_name in tqdm(enumerate(task_names), total=len(task_names), desc='Normalize labels'):
            # skip prefix fr_
            if task_name.startswith('fr_'):
                continue
            scaler = StandardScaler()

            # use mask
            scaler.fit(train_labels[:, i][train_labels_mask[:, i]].reshape(-1, 1))
            # use mask
            train_labels[:, i][train_labels_mask[:, i]] = torch.tensor(scaler.transform(train_labels[:, i][train_labels_mask[:, i]].reshape(-1, 1)).reshape(-1), dtype=torch.float32)
            # use mask
            val_labels[:, i][val_labels_mask[:, i]] = torch.tensor(scaler.transform(val_labels[:, i][val_labels_mask[:, i]].reshape(-1, 1)).reshape(-1), dtype=torch.float32)
            scaler_list.append(scaler)

        # assign labels
        for i, ele in enumerate(train_data):
            ele[2] = train_labels[i]
        
        for i, ele in enumerate(val_data):
            ele[2] = val_labels[i]

        
        self.train_dataset = MultiTaskPreTrainingDataset(train_data, smi_tokenizer)
        self.val_dataset = MultiTaskPreTrainingDataset(val_data, smi_tokenizer)

        self.collator = CustomDataCollatorForMTP(
            tokenizer=None,
            txt_tokenizer=txt_tokenizer,
            smi_tokenizer=smi_tokenizer, 
            mlm_probability=0.15,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator
        )


if __name__ == "__main__":
    file_name = 'multi-task-pretrain.pt'
    txt_tokenizer = load_tokenizer('PubMedBERT')
    smi_tokenizer = load_tokenizer('CHEM-BERT')

    dm = MultiTaskPreTrainingDataModule(
        file_name=file_name,
        txt_tokenizer=txt_tokenizer,
        smi_tokenizer=smi_tokenizer,
        batch_size=2,
    )

    train_loader = dm.train_dataloader()

    for batch in train_loader:
        print(batch)
        break
    