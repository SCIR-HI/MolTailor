import sys
from pathlib import Path, PosixPath
from typing import Union, Any, Tuple, Dict, List

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
data_dir = base_dir / 'data'

import json

import torch

from torch.utils.data import Dataset
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from models.load import load_tokenizer
from models.chembert.tokenizer import ChemBertTokenizer


class MultiTaskPreTrainingDataset(Dataset):
    def __init__(self, data: list, smi_tokenizer: ChemBertTokenizer) -> None:
        super().__init__()
        self.data = data

        self.smi_tokenizer = smi_tokenizer

    def __getitem__(self, index) -> Any:
        data = self.data[index]
        return {
            'smiles': data[0],
            'task_description': data[1],
            'labels-mtr': data[2],
            'labels-mtr-mask': data[3],
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
if __name__ == '__main__':
    file_name = 'multi-task-pretrain.pt'
    data = torch.load(data_dir / file_name)
    txt_tokenizer = load_tokenizer('PubMedBERT')
    smi_tokenizer = load_tokenizer('CHEM-BERT')
    dataset = MultiTaskPreTrainingDataset(data, smi_tokenizer)

    # import pandas as pd
    # df = pd.read_json('/home/hqguo/workspace/01-st/pretrain-dual/data/clean/descriptors.jsonl', lines=True)
    # df.describe()
    # column_name_list = list(df.columns)
    # column_name_list.pop(0)
    # print(column_name_list)

    # save
    # import json
    # with open(data_dir / 'task_names.json', 'w') as f:
    #     json.dump(column_name_list, f)

    # check if nan in data[i]['labels-mtr']
    # from tqdm import tqdm
    # for i in tqdm(range(len(data)), total=len(data)):
    #     if torch.isnan(data[i]['txt']['labels-mtr']).any():
    #         print(i)
    #         break
