import sys
from pathlib import Path, PosixPath

base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))

import torch

from torch import nn

from pytorch_lightning import LightningModule


class MoLFormer(LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.hparams = config
        self.mode = config.mode
        self.save_hyperparameters(config)

        self.tokenizer = tokenizer

        n_vocab = len(tokenizer.vocab), config.n_embd

        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
