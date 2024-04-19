import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
data_dir = base_dir / 'data'


import json

import torch
import pandas as pd
import numpy as np

from tqdm import tqdm

# from models.load import load_tokenizer


def remove_overlap_smiles():
    data = pd.read_json(data_dir / 'mt-mtr-origin.jsonl', lines=True)
    # molenet_smiles.json contains SMILES from eight tasks in MoleculeNet used in the paper
    # And the SMILES in this file have been validated, normalized, and deduplicated.
    # BBBP, ClinTox, HIV, Tox21, ESOL, FreeSolv, Lip, QM8
    with open(data_dir / 'temporary/molnet_smiles.json', 'r') as f:
        molnet_smiles = json.load(f)
    
    # keep data that smiles not in molnet_smiles
    data = data[data['smiles'].isin(molnet_smiles) == False]
    print(f'len(data): {len(data)}')
    data.to_json(data_dir / 'mt-mtr-origin-clean.jsonl', orient='records', lines=True)


# build mt-mtr pretrain data
def generate_mtr_data():
    with open(data_dir / 'temporary/task_names.json', 'r') as f:
        task_names = json.load(f)
        num_tasks = len(task_names)

    data = pd.read_json(data_dir / 'mt-mtr-origin.jsonl', lines=True)
    
    nan_num = 0
    valid_num = 0
    too_large_num = 0
    mtr_task_list = []

    for idx, item in tqdm(data.iterrows(), total=len(data)):
        mtr_labels = torch.zeros(num_tasks, dtype=torch.float32)
        mtr_masks = torch.zeros(num_tasks, dtype=torch.bool)
        descriptors = json.loads(item['descriptors'])

        for j, task_name in enumerate(task_names):
            if task_name in descriptors.keys():
                if np.isnan(descriptors[task_name]):
                    nan_num += 1
                    continue
                if abs(descriptors[task_name]) > 1e5:
                    too_large_num += 1
                    continue
                valid_num += 1
                tensor = torch.tensor(descriptors[task_name], dtype=torch.float32)
                assert not (torch.isinf(tensor) or torch.isnan(tensor)), f'unexpected value: {tensor}'
                mtr_labels[j] = tensor
                mtr_masks[j] = True
        mtr_task_list.append([item['smiles'] , item['task_description'], mtr_labels, mtr_masks])
        
    print(f'nan_num: {nan_num}')
    print(f'too_large_num: {too_large_num}')
    print(f'valid_num: {valid_num}')

    torch.save(mtr_task_list, data_dir / 'mt-mtr.pt')


if __name__ == '__main__':
    remove_overlap_smiles()
    generate_mtr_data()