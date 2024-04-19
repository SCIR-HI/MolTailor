import sys

import numpy as np
import pandas as pd

from pathlib import Path
from rdkit.Chem import AllChem
from icecream import ic
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
data_dir = base_dir / 'data'


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


from rdkit.Chem import Descriptors
desc_list = Descriptors.descList
# for desc in desc_list:
#     print(desc[0])
# print(len(desc_list))

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


def preprocess(data_path: str, task_name: str, col_smiles: str = 'smiles', col_names: list=['smiles', 'p_np']) -> None:
    data_path = Path(data_path).resolve()
    
    df = pd.read_csv(data_path)
    ic(f'len of {task_name}: {df.shape[0]}')
    
    tqdm.pandas()
    df.loc[:, 'smiles'] = df[col_smiles].progress_apply(clean_smiles)
    
    df_cleaned = df[df['smiles']!='-']
    ic(f'len of cleaned {task_name}: {df_cleaned.shape[0]}')
    
    df_cleaned = df_cleaned[col_names]

    save_path = data_path.parent.parent / (task_name + data_path.suffix)
    
    df_cleaned.to_csv(save_path, index=False)


if __name__ == '__main__':
    task2file = {
        'bbbp': 'BBBP.csv',
        'clintox': 'clintox.csv',
        'hiv': 'HIV.csv',
        'tox21': 'tox21.csv',
        'esol': 'delaney-processed.csv',
        'freesolv': 'SAMPL.csv',
        'lipophilicity': 'Lipophilicity.csv',
        'qm7': 'qm7.csv',
        'qm8': 'qm8.csv',
        'qm9': 'qm9.csv'
    }

    task2smiles = {
        'bbbp': 'smiles',
        'clintox': 'smiles',
        'hiv': 'smiles',
        'tox21': 'smiles',
        'esol': 'smiles',
        'freesolv': 'smiles',
        'lipophilicity': 'smiles',
        'qm7': 'smiles',
        'qm8': 'smiles',
        'qm9': 'smiles'
    }

    task2cols = {
        'bbbp': ['smiles', 'p_np'],
        'clintox': ['smiles', 'FDA_APPROVED', 'CT_TOX'],
        'hiv': ['smiles', 'HIV_active'],
        'tox21': [
            'smiles', 
            'NR-AR', 
            'NR-AR-LBD', 
            'NR-AhR', 
            'NR-Aromatase', 
            'NR-ER', 
            'NR-ER-LBD', 
            'NR-PPAR-gamma', 
            'SR-ARE', 
            'SR-ATAD5', 
            'SR-HSE', 
            'SR-MMP', 
            'SR-p53'
        ],
        'esol': ['smiles', 'measured log solubility in mols per litre'],
        'freesolv': ['smiles', 'expt'],
        'lipophilicity': ['smiles', 'exp'],
        'qm7': ['smiles', 'u0_atom'],
        'qm8': ['smiles',
                'E1-CC2',
                'E2-CC2',
                'f1-CC2',
                'f2-CC2',
                'E1-PBE0',
                'E2-PBE0',
                'f1-PBE0',
                'f2-PBE0',
                'E1-PBE0.1',
                'E2-PBE0.1',
                'f1-PBE0.1',
                'f2-PBE0.1',
                'E1-CAM',
                'E2-CAM',
                'f1-CAM',
                'f2-CAM'],
        'qm9': [
            'smiles', 
            'mu',
            'alpha',
            'homo',
            'lumo',
            'gap',
            'r2',
            'zpve',
            'u0',
            'u298',
            'h298',
            'g298',
            'cv',
        ]
    }

    # preprocess
    for task_name in ['bbbp', 'clintox', 'hiv', 'tox21', 'esol', 'freesolv', 'lipophilicity', 'qm8']:
        data_path = data_dir / 'raw' / task2file[task_name]
        preprocess(data_path, task_name, col_smiles=task2smiles[task_name], col_names=task2cols[task_name])

