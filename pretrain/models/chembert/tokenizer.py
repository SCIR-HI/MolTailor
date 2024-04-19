import re
import json

from typing import List, Dict, Tuple
from collections import defaultdict

import torch
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdmolops
from torch.nn.utils.rnn import pad_sequence


class ChemBertTokenizer:
    """_summary_
    """
    def __init__(self, vocab_path, max_seq_len=256, add_special_token=True, truncation=True) -> None:
        """_summary_

        Args:
            vocab_path (_type_): _description_
            max_seq_len (int, optional): _description_. Defaults to 256.
            add_special_token (bool, optional): _description_. Defaults to True.
            truncation (int, optional): Ture-表示需要截断, False-表示不需要截断. Defaults to 1.
        """
        self.atom_vocab = ['C', 'O', 'n', 'c', 'F', 'N', 'S', 's', 'o', 'P', 'R', 'L', 'X', 'B', 'I', 'i', 'p', 'A']
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        # reverse
        self.reve_vocab         = dict(zip(self.vocab.values(), self.vocab.keys()))
        self.max_seq_len        = max_seq_len
        self.pad_id             = self.vocab['<pad>']
        self.mask_id            = self.vocab['<mask>']
        self.unk_id             = self.vocab['<unk>']
        self.start_id           = self.vocab['<start>']
        self.end_id             = self.vocab['<end>']
        self.add_special_token  = add_special_token
        self.truncation         = truncation

    
    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def replace_halogen(self, string):
        """Regex to replace Br and Cl with single letters"""
        br = re.compile('Br')
        cl = re.compile('Cl')
        sn = re.compile('Sn')
        na = re.compile('Na')
        string = br.sub('R', string)
        string = cl.sub('L', string)
        string = sn.sub('X', string)
        string = na.sub('A', string)
        return string

    def zero_padding(self, array, shape):
        if array.shape[0] > shape[0]:
            array = array[:shape[0],:shape[1]]
        padded = np.zeros(shape, dtype=np.float32)
        padded[:array.shape[0], :array.shape[1]] = array
        return padded
        
    
    def char2num(self, smiles):
        """将经过replace halogen函数处理后的smiles转换为对应的ids

        Args:
            smiles (str): 将smiles先经过replace halogen函数将连个字符组成的元素替换为一个字符

        Returns:
            idx_list: 切分后的id列表
            adj_mask: 邻接矩阵的掩模列表
        """
        tokens = [i for i in smiles]
        idx_list = []
        adj_mask = []

        for i, token in enumerate(tokens):
            if token in self.atom_vocab:
                adj_mask.append(1)
            else:
                adj_mask.append(0)
            idx_list.append(self.vocab.get(token, self.unk_id))

        return idx_list, adj_mask
    
    def num2char(self, idx_list):
        token_list = []
        for idx in idx_list:
            token = self.reve_vocab.get(idx, -1)
            assert token!=-1, "contain unknown id"
            if token == "R":
                token = "Br"
            elif token == "L":
                token = "Cl"
            elif token == "X":
                token = "Sn"
            elif token == "A":
                token = "Na"
            token_list.append(token)
        return ''.join(token_list)
    
    def encode(self, smiles):
        smiles_rpl = self.replace_halogen(smiles)
        smiles_mol = Chem.MolFromSmiles(smiles)
        
        idx_list, adj_mask = self.char2num(smiles_rpl)
        
        if smiles_mol:
            adj_matx = rdmolops.GetAdjacencyMatrix(smiles_mol)
        else:
            adj_matx = np.zeros((1, 1), dtype=np.float32)
        
        if self.truncation:
            idx_list = idx_list[:(self.max_seq_len-2)]
            adj_mask = adj_mask[:(self.max_seq_len-2)]
        
        if self.add_special_token:
            idx_list = [self.start_id] + idx_list + [self.end_id]
            adj_mask = [0] + adj_mask + [0]
            

        adj_matx = adj_matx[:(self.max_seq_len-2), :(self.max_seq_len-2)]
        
        padded  = np.zeros((self.max_seq_len, self.max_seq_len), dtype=np.float32)
        padded[1:(adj_matx.shape[0]+1), 1:(adj_matx.shape[1]+1)] = adj_matx # fix
        # padded[:(adj_matx.shape[0]), :(adj_matx.shape[1])] = adj_matx # origin
        adj_matx  = padded

        adj_matx = adj_matx.tolist()

        idx_mask = [0] * len(idx_list)
            
        return idx_list, idx_mask, adj_mask, adj_matx

    def pad(self, tokenized_list, return_special_tokens_mask=False):
        assert type(tokenized_list) == list, "batch must be a list"
        keys = tokenized_list[0].keys()
        assert 'input' in keys, "input must be in batch"
        assert 'imask' in keys, "imask must be in batch"
        assert 'amask' in keys, "amask must be in batch"
        assert 'amatx' in keys, "amatx must be in batch"

        batch = defaultdict(list)
        for ele in tokenized_list:
            for key in keys:
                batch[key].append(ele[key])
        
        batch['input'] = pad_sequence(batch['input'], batch_first=True, padding_value=self.pad_id)
        batch['imask'] = pad_sequence(batch['imask'], batch_first=True, padding_value=1)
        batch['amask'] = pad_sequence(batch['amask'], batch_first=True, padding_value=0)
        batch['amatx'] = torch.cat([ele.unsqueeze(0) for ele in batch['amatx']])
        if return_special_tokens_mask:
            # special tokens: start, end, pad
            batch['special_tokens_mask'] = batch['input'].eq(self.start_id) + batch['input'].eq(self.end_id) + batch['input'].eq(self.pad_id)

        return dict(batch)
    
    def batch_encode_plus(self, smiles_list, return_special_tokens_mask=False):
        tokenized_list = []
        for smiles in smiles_list:
            input, imask, amask, amatx = self.encode(smiles)
            tokenized_list.append(
                {
                    'input': torch.tensor(input, dtype=torch.long),
                    'imask': torch.tensor(imask, dtype=torch.bool),
                    'amask': torch.tensor(amask, dtype=torch.float),
                    'amatx': torch.tensor(amatx, dtype=torch.float),
                }
            )
        return self.pad(tokenized_list, return_special_tokens_mask=return_special_tokens_mask)

    
    def decode(self, idx_list):
        return self.num2char(idx_list)
