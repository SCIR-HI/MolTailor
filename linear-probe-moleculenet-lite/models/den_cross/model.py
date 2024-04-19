'''
Author: simwit 517992857@qq.com
Date: 2023-07-29 20:41:17
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-07-30 10:25:30
FilePath: /workspace/01-st/finetune-moleculenet/models/chembert/model.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''


import torch
import torch.nn as nn
try:
    from .embedding import Smiles_embedding
except ImportError:
    from embedding import Smiles_embedding

class ChemBert(nn.Module):
    def __init__(self, vocab_size, max_len=256, feature_dim=1024, nhead=4, feedforward_dim=1024, nlayers=6, adj=False, dropout_rate=0):
        super(ChemBert, self).__init__()
        self.embedding = Smiles_embedding(vocab_size, feature_dim, max_len, adj=adj)
        trans_layer = nn.TransformerEncoderLayer(feature_dim, nhead, feedforward_dim, activation='gelu', dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(trans_layer, nlayers)

        #self.linear = Masked_prediction(feedforward_dim, vocab_size)

    def forward(self, input, imask=None, amask=None, amatx=None):
        # True -> masking on zero-padding. False -> do nothing
        #mask = (src == 0).unsqueeze(1).repeat(1, src.size(1), 1).unsqueeze(1)
        # if imask is None:
            # imask = (input == 0)
            # imask = imask.type(torch.bool)
        #print(mask.shape)
        
        pos_num = torch.arange(input.size(1)).repeat(input.size(0),1).to(input.device)

        x = self.embedding(input, pos_num, amask, amatx)
        x = self.transformer_encoder(x.transpose(1,0), src_key_padding_mask=imask)
        x = x.transpose(1,0)
        #x = self.linear(x)
        return x