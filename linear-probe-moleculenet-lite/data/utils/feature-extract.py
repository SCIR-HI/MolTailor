import os
import sys
import json
import argparse

import torch
import dgl
import numpy as np
import pandas as pd

from pathlib import Path
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from icecream import ic
from tqdm import tqdm
from pytorch_lightning import seed_everything

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
data_dir = base_dir / 'data'

from models.load import load_model, load_tokenizer


from rdkit.Chem import Descriptors
desc_list = Descriptors.descList


def get_desc(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return np.random.uniform(-1, 1, size=(1, 209)).astype(np.float32)
    desc = []
    for _, desc_func in desc_list:
        try:
            value = desc_func(mol)
            # determine if too large
            if (abs(value) > 1e+5) or np.isinf(value) or np.isnan(value):
                value = 0
            desc.append(value)
        except:
            desc.append(0)

    return np.asarray(desc, dtype=np.float32).reshape(1, -1)


def feature_extract(task_name: str, model_name: str, device: str='cuda', prompt_file: str = 'prompt4molnet.json') -> None:
    df = pd.read_csv(data_dir / (task_name + '.csv'))
    
    if model_name == 'Random':
        feature_list = []
        for _, serise in tqdm(df.iterrows(), total=df.shape[0]):
            # transform to float 32 (default: float64)
            feature = np.random.uniform(-1, 1, size=(1, 768)).astype(np.float32)
            feature_list.append(feature)
    
    elif model_name == 'RDKit-FP':
        feature_list = []
        for _, serise in tqdm(df.iterrows(), total=df.shape[0]): 
            mol = AllChem.MolFromSmiles(serise['smiles'])
            # RDkit-FP
            fp = AllChem.RDKFingerprint(mol)
            feature = np.zeros((1,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, feature)
            feature_list.append(feature.reshape(1, -1))
    
    elif model_name == 'Morgan-FP':
        feature_list = []
        for _, serise in tqdm(df.iterrows(), total=df.shape[0]):
            mol = AllChem.MolFromSmiles(serise['smiles'])
            # Morgan-FP
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
            feature = np.zeros((1,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, feature)
            feature_list.append(feature.reshape(1, -1))
    
    elif model_name == 'MACCS-FP':
        feature_list = []
        for _, serise in tqdm(df.iterrows(), total=df.shape[0]):
            mol = AllChem.MolFromSmiles(serise['smiles'])
            # MACCS-FP
            fp = MACCSkeys.GenMACCSKeys(mol)
            feature = np.zeros((1,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, feature)
            feature_list.append(feature.reshape(1, -1))
    
    elif model_name == 'RDKit-DP':
        feature_list = []
        for _, serise in tqdm(df.iterrows(), total=df.shape[0]): 
            feature = get_desc(serise['smiles'])
            feature_list.append(feature)

    elif model_name == 'KCL':
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name).to(device)
        model.eval()

        feature_list = []
        for _, serise in tqdm(df.iterrows(), total=df.shape[0]):
            graph = tokenizer(serise['smiles'])
            if graph is None:
                feature_list.append(np.random.uniform(-1, 1, size=(1, 128)).astype(np.float32))
                print('graph is None')
            else:
                try:
                    dg = dgl.batch([graph]).to(device)
                    with torch.no_grad():
                        feature = model(dg).detach().cpu().numpy()
                except:
                    feature = np.random.uniform(-1, 1, size=(1, 128)).astype(np.float32)
                    print('error')
                feature_list.append(feature)

    elif (model_name == 'Grover') or (model_name == 'Grover-Base'):
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name).to(device)
        model.eval()

        feature_list = []
        for _, serise in tqdm(df.iterrows(), total=df.shape[0]):
            batch = tokenizer([serise['smiles']])
            batch = batch.get_components()
            batch = [ele.to(device) for ele in batch]
            with torch.no_grad():
                feature = model(batch, features_batch=[None]).detach().cpu().numpy()
            feature_list.append(feature)
    
    elif model_name == 'MolCLR':
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name).to(device)
        model.eval()

        feature_list = []
        for i, serise in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                tokenized = tokenizer(serise['smiles']).to(device)
                with torch.no_grad():
                    feature = model(tokenized).detach().cpu().numpy()
            except:
                print(f'error {i}')
                feature = np.random.uniform(-1, 1, size=(1, 512)).astype(np.float32)
            feature_list.append(feature)
    
    elif model_name == 'MoMu':
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name).to(device)
        model.eval()

        feature_list = []
        for i, serise in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                tokenized = tokenizer(serise['smiles']).to(device)
                with torch.no_grad():
                    feature = model(tokenized).detach().cpu().numpy()
            except:
                print(f'error {i}')
                feature = np.random.uniform(-1, 1, size=(1, 300)).astype(np.float32)
            feature_list.append(feature)

    elif model_name == 'CLAMP':
        model = load_model(model_name).to(device)
        model.eval()

        feature_list = []
        for _, serise in tqdm(df.iterrows(), total=df.shape[0]):
            with torch.no_grad():
                feature = model.encode_smiles([serise['smiles']]).detach().cpu().numpy()
            feature_list.append(feature)

    # elif model_name == 'Uni-Mol':
    #     model = load_model(model_name)
    #     smiles_list = df['smiles'].tolist()
    #     repr_dict = model.get_repr(smiles_list)
    #     feature = np.array(repr_dict['cls_repr'], dtype=np.float32)
    #     print(feature.shape)
    #     with open(data_dir / 'feature' / (f'{task_name}-{model_name}' + '.npy'), 'wb') as f:
    #         np.save(f, feature)
    #     return
    
    elif 'Mole-BERT' == model_name:
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name).to(device)
        model.eval()

        feature_list = []
        for i, serise in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                tokenized = tokenizer(serise['smiles']).to(device)
                with torch.no_grad():
                    feature = model(tokenized).detach().cpu().numpy()
            except:
                print(f'error {i}')
                feature = np.random.uniform(-1, 1, size=(1, 300)).astype(np.float32)
            feature_list.append(feature) 
    
    elif 'CHEM-BERT' == model_name:
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name).to(device)
        model.eval()
        
        feature_list = []
        for _, serise in tqdm(df.iterrows(), total=df.shape[0]):
            input, imask, amask, amatx = tokenizer.encode(serise['smiles'])
            with torch.no_grad():
                feature = model(
                    torch.tensor(input, dtype=torch.long).unsqueeze(0).to(device),
                    torch.tensor(imask, dtype=torch.bool).unsqueeze(0).to(device),
                    torch.tensor(amask, dtype=torch.float).unsqueeze(0).to(device),
                    torch.tensor(amatx, dtype=torch.float).unsqueeze(0).to(device),         
                )[:, 0, :].detach().cpu().numpy()
            feature_list.append(feature)
    
    elif model_name in ['BERT', 'RoBERTa', 'SciBERT', 'PubMedBERT', 'BioLinkBERT', 'ChemBERTa-77M-MTR', 'ChemBERTa-10M-MTR', 'ChemBERTa-77M-MLM']:
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name).to(device)
        model.eval()
        
        feature_list = []
        for _, serise in tqdm(df.iterrows(), total=df.shape[0]):
            tokenized = tokenizer.encode(serise['smiles'], return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                feature = model(tokenized).last_hidden_state[:, 0, :].detach().cpu().numpy()
            feature_list.append(feature)

    elif model_name in ['MolT5', 'T5', 'T5-MEAN', 'MolT5-MEAN','TCT5-MEAN', 'TCT5']:
        model_name_copy = model_name
        mean_flag = False
        if 'MEAN' in model_name:
            mean_flag = True
            model_name = model_name.split('-')[0]
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name).to(device)
        model.eval()

        feature_list = []
        for _, serise in tqdm(df.iterrows(), total=df.shape[0]):
            tokenized = tokenizer(serise['smiles'], return_tensors='pt').to(device)
            with torch.no_grad():
                if mean_flag:
                    feature = model.encoder(**tokenized).last_hidden_state.mean(dim=1).detach().cpu().numpy()
                else:
                    outputs = model.encoder(**tokenized)
                    end_token_idx = (tokenized['input_ids'] == tokenizer.eos_token_id).nonzero(as_tuple=True)[1]
                    # use eos token as feature
                    feature = outputs.last_hidden_state[0, end_token_idx, :].detach().cpu().numpy()
            feature_list.append(feature)
        model_name = model_name_copy
    
    elif 'DEN-ChemBERTa' in model_name:
        model_name_copy = model_name
        ckpt_id = model_name_copy.split('-')[-1]
        print(f'DEN-ChemBERTa ckpt_id for molnet: {ckpt_id}')
        model_name = 'DEN-ChemBERTa'
        model = load_model(model_name, ckpt_id=ckpt_id).to(device)
        tokenizer_smi, tokenizer_txt = load_tokenizer(model_name)
        model.eval()
        with open(data_dir / prompt_file, 'r') as f:
            prompt = json.load(f)[task_name]
            print(f'use prompt: {prompt_file}')
        tokenized_txt = tokenizer_txt(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        tokenized_txt = {
            'input': tokenized_txt['input_ids'],
            'imask': tokenized_txt['attention_mask'],
        }

        feature_list = []
        for _, serise in tqdm(df.iterrows(), total=df.shape[0]):
            tokenized_smi = tokenizer_smi(serise['smiles'], return_tensors='pt', truncation=True, max_length=512).to(device)
            data = {
                'txt': tokenized_txt,
                'smi': tokenized_smi
            }
            with torch.no_grad():
                feature = model(txt=data['txt'], smi=data['smi'], return_dict=True).last_hidden_state_txt[:, 0, :].detach().cpu().numpy()
            feature_list.append(feature)
        model_name = model_name_copy

    # elif model_name in ['DEN', 'DEN-1', 'DEN-6nn1f40f', 'DEN-l3x8us5x', 'DEN-o6d6inqz', 'DEN-f9x97q2q']:
    elif 'DEN' in model_name:
        model_name_copy = model_name
        ckpt_id = model_name_copy.split('-')[-1]
        print(f'ckpt_id for molnet: {ckpt_id}')
        model_name = 'DEN'
        model = load_model(model_name, ckpt_id=ckpt_id).to(device)
        tokenizer_smi, tokenizer_txt = load_tokenizer(model_name)
        model.eval()
        with open(data_dir / prompt_file, 'r') as f:
            prompt = json.load(f)[task_name]
            print(f'use prompt: {prompt_file}')
        tokenized_txt = tokenizer_txt(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        tokenized_txt = {
            'input': tokenized_txt['input_ids'],
            'imask': tokenized_txt['attention_mask'],
        }

        feature_list = []
        for _, serise in tqdm(df.iterrows(), total=df.shape[0]):
            input, imask, amask, amatx = tokenizer_smi.encode(serise['smiles'])
            data = {
                'txt': tokenized_txt,
                'smi': {
                    'input': torch.tensor(input, dtype=torch.long).unsqueeze(0).to(device),
                    'imask': torch.tensor(imask, dtype=torch.bool).unsqueeze(0).to(device),
                    'amask': torch.tensor(amask, dtype=torch.float).unsqueeze(0).to(device),
                    'amatx': torch.tensor(amatx, dtype=torch.float).unsqueeze(0).to(device),  
                }
            }
            with torch.no_grad():
                feature = model(txt=data['txt'], smi=data['smi'], return_dict=True).last_hidden_state_txt[:, 0, :].detach().cpu().numpy()
            feature_list.append(feature)
        model_name = model_name_copy

    else:
        print(f'Wrong model_name: {model_name}!')
        return
    
    feature = np.concatenate(feature_list, axis=0)
    print(feature.shape)
    with open(data_dir / 'feature' / (f'{task_name}-{model_name}' + '.npy'), 'wb') as f:
        np.save(f, feature)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Random')
    parser.add_argument('--prompt_file', type=str, default='prompt4molnet.json')
    parser.add_argument('--task_name', type=str, default='')
    parser.add_argument('--ckpt_id', type=str, default='')
    args = parser.parse_args()
    model_name = args.model_name

    model_list = [
        'Random',
        'RDKit-FP',
        'Morgan-FP',
        'MACCS-FP',
        'RDKit-DP',
        'KCL',
        'Grover',
        'MolCLR',
        'MoMu',
        'CLAMP',
        'Uni-Mol',
        'Mole-BERT',
        'CHEM-BERT',
        'BERT', 
        'RoBERTa', 
        'SciBERT', 
        'PubMedBERT', 
        'BioLinkBERT', 
        'ChemBERTa-77M-MTR', 
        'ChemBERTa-10M-MTR', 
        'ChemBERTa-77M-MLM',
        'MolT5', 
        'T5', 
        'TCT5'
        'DEN-ChemBERTa', 'u02pzsl2', '0al3aezz'
        'DEN', 'f9x97q2q', 
    ]

    # multi tasks
    print(f'Feature extract: {model_name}')

    if args.task_name:
        task_name = args.task_name
        assert task_name in ['bbbp', 'clintox', 'hiv', 'tox21', 'esol', 'freesolv', 'lipophilicity', 'qm8'], "wrong task_name!"
        print(f'preprocess {task_name}')
        seed_everything(0)
        feature_extract(task_name, model_name, prompt_file=args.prompt_file)
    else:
        for task_name in ['bbbp', 'clintox', 'hiv', 'tox21', 'esol', 'freesolv', 'lipophilicity', 'qm8']:
            print(f'preprocess {task_name}')
            seed_everything(0)
            feature_extract(task_name, model_name, prompt_file=args.prompt_file)


