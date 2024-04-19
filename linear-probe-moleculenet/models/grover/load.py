import sys

from pathlib import Path, PosixPath
from types import SimpleNamespace

try:
    from grover import Grover
    from molgraph import mol2graph
except:
    from .grover import Grover
    from .molgraph import mol2graph

base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/Grover'

import json
import torch

def load_grover(path: PosixPath = model_path, type='large'):
    if type == 'large':
        ckpt = torch.load(path / 'grover_large.pt')
    elif type == 'base':
        ckpt = torch.load(path / 'grover_base.pt')

    args = ckpt['args']
    args.cuda = True
    args.dropout = 0.0
    args.features_only = False
    args.features_dim = 0

    model = Grover(args)

    # add missing keys
    # readout.cached_zero_vector -> nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
    ckpt['state_dict']['readout.cached_zero_vector'] = torch.zeros(args.hidden_size)

    model.load_state_dict(ckpt['state_dict'])

    return model


def load_grover_tokenizer(path: PosixPath = model_path):
    ckpt = torch.load(path / 'grover_large.pt')

    args = ckpt['args']
    args.cuda = True
    args.dropout = 0.0
    args.features_only = False
    args.features_dim = 0
    args.no_cache = True

    def tokenizer(smiles_batch, shared_dict={}, args=args):
        return mol2graph(smiles_batch, args=args, shared_dict=shared_dict)

    return tokenizer

if __name__ == '__main__':
    model = load_grover()
    print(model)
    model.to('cuda')

    tokenizer = load_grover_tokenizer()
    batch = tokenizer(['CC', 'CCC'])
    # print(batch)
    batch = batch.get_components()
    # print(batch)
    batch = [ele.to('cuda') for ele in batch]
    output = model(batch, features_batch=[None])
    print(output)
    print(output.shape)

    
