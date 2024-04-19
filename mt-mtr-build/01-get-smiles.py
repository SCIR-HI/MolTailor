import sys

import requests
import pandas as pd

from pathlib import Path
from rdkit.Chem import AllChem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

base_dir = Path(__file__).resolve().parent
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


def cas2smiles(cas: str):
    # https://cactus.nci.nih.gov/chemical/structure
    url = f'https://cactus.nci.nih.gov/chemical/structure/{cas}/smiles'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(response.status_code)
            print(response.text)
            return ''
    except Exception as e:
        print(e)
        return ''


if __name__ == '__main__':

    # load smiles from chebi and drugbank which has been cleaned using the functions above    
    chebi_df = pd.read_json(data_dir / 'temporary/smiles-chebi.jsonl', orient='records', lines=True)
    drugbank_df = pd.read_json(data_dir / 'temporary/smiles-drugbank.jsonl', orient='records', lines=True)

    print(f'chebi_df: {chebi_df.shape}')
    print(f'drugbank_df: {drugbank_df.shape}')

    df = pd.concat([chebi_df, drugbank_df], axis=0)
    print(df.shape)

    df = df.drop_duplicates(subset=['smiles'])
    print(df.shape)

    df.to_json(data_dir / 'temporary/smiles.jsonl', orient='records', lines=True)