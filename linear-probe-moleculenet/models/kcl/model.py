import sys
import json
from typing import Any
import torch
import pickle

from pathlib import Path, PosixPath

try:
    from encoder import KMPNNGNN
    from layer import Set2Set
except ImportError:
    from .encoder import KMPNNGNN
    from .layer import Set2Set

import pytorch_lightning as pl

class KCL(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        model_path = config['model_path']

        loaded_dict = pickle.load(open(model_path / config['initial_name'], 'rb'))
        entity_emb, relation_emb = loaded_dict['entity_emb'], loaded_dict['relation_emb']
        
        self.encoder = KMPNNGNN(config, entity_emb, relation_emb)
        self.encoder.load_state_dict(torch.load(model_path / config['encoder_name']))
        self.readout = Set2Set(self.encoder.out_dim, n_iters=6, n_layers=3)
        self.readout.load_state_dict(torch.load(model_path / config['readout_name']))

    def forward(self, batch) -> Any:
        bg = batch
        graph_embedding = self.readout(bg, self.encoder(bg))

        return graph_embedding