import random

from collections import defaultdict
from typing import List, Tuple, Union, Dict, Set

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A smiles string or an RDKit molecule.
    :param include_chirality: Whether to include chirality.
    :return:
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def smiles_to_scaffold(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def scaffold_split(data: List[str],
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = False) -> Tuple[List[int], List[int], List[int]]:
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param data: A MoleculeDataset.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
    :param seed: Seed for shuffling when doing balanced splitting.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert sum(sizes) == 1
    
    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = smiles_to_scaffold(data, use_indices=True)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = list(scaffold_to_indices.values())
        random.shuffle(index_sets)
        index_sets = sorted(index_sets,
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    print(f'Total scaffolds = {len(scaffold_to_indices):,} | '
        f'train scaffolds = {train_scaffold_count:,} | '
        f'val scaffolds = {val_scaffold_count:,} | '
        f'test scaffolds = {test_scaffold_count:,}')


    return train, val, test


def random_split(data: List[str],
                 sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 ) -> Tuple[List[int], List[int], List[int]]:
    
    data_size = len(data)
    index_list = list(range(data_size))
    random.shuffle(index_list)

    train_size = int(sizes[0] * data_size)
    train_val_size = int((sizes[0] + sizes[1]) * data_size)

    train = index_list[:train_size]
    val = index_list[train_size:train_val_size]
    test = index_list[train_val_size:]
    
    return train, val, test


def split_data(data: List[str], 
                sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1), 
                split_type: str = 'scaffold', 
                balanced: bool = False) -> Tuple[List[int], List[int], List[int]]:
    if split_type == 'scaffold':
        return scaffold_split(data, sizes, balanced)
    elif split_type == 'random':
        return random_split(data, sizes)
    else:
        raise ValueError(f'Invalid split type "{split_type}"')