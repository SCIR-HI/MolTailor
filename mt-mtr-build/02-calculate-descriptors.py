import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
data_dir = base_dir / 'data'

import pandas as pd

from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem import Descriptors


RDLogger.DisableLog('rdApp.*')
desc_list = Descriptors.descList


def get_desc(smiles: str):
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc = {
        'smiles': smiles,
    }
    for desc_name, desc_func in desc_list:
        try:
            desc[desc_name] = desc_func(mol)
        except:
            desc[desc_name] = None
    return desc


def clean_smiles(smiles) -> str:
    mol = AllChem.MolFromSmiles(smiles)
    
    if mol is None:
        return '-'        
    if mol.GetNumHeavyAtoms() == 0:
        return '-'
    try:
        AllChem.SanitizeMol(mol)
    except:
        return '-'
    canonical_smiles = AllChem.MolToSmiles(mol)
    
    return canonical_smiles


def calculate_ratio(df_desc: pd.DataFrame):
    # clean smiles
    tqdm.pandas()
    df_desc['smiles'] = df_desc['smiles'].progress_apply(clean_smiles)

    # remove empty smiles
    df_desc = df_desc[df_desc['smiles'] != '-']
    df_desc.shape


    # statistic the zero num and non-zero num for fr_ prefix descriptors
    # fr_ prefix descriptors
    fr_desc_list = [desc for desc in desc_list if desc[0].startswith('fr_')]

    # zero num
    zero_num = []
    for desc_name, _ in tqdm(fr_desc_list):
        zero_num.append((desc_name, (df_desc[desc_name] == 0).sum()))

    zero_num = sorted(zero_num, key=lambda x: x[1], reverse=True)

    # ratio
    ratio = []

    for desc_name, _ in tqdm(fr_desc_list):
        ratio.append((desc_name, (df_desc[desc_name] == 0).sum() / len(df_desc)))

    ratio = sorted(ratio, key=lambda x: x[1], reverse=True)

    # save ratio in json file
    import json

    with open(data_dir / 'temporary/ratio.json', 'w') as f:
        ratio = dict(ratio)
        json.dump(ratio, f, indent=4)

    # load ratio from json file
    with open(data_dir / 'temporary/ratio.json', 'r') as f:
        ratio = json.load(f)

    print(ratio)


if __name__ == '__main__':

    df = pd.read_json(data_dir / 'temporary/smiles.jsonl', lines=True)

    descs = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row['smiles']
        desc = get_desc(smiles)
        descs.append(desc)

    descs = [desc for desc in descs if desc is not None]
    df_desc = pd.DataFrame(descs)
    df_desc.shape
    df_desc.to_json(data_dir / 'temporary/descriptors.jsonl', lines=True, orient='records')
