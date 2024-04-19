'''
Author: simwit 517992857@qq.com
Date: 2023-08-07 09:47:52
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-11 21:54:39
FilePath: /hqguo/workspace/01-st/finetune-moleculenet/models/den/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
import json
import torch

try:
    from .config import DenConfig
    from .den import DenForFineTuning
    from .model import ChemBert
    from .tokenizer import ChemBertTokenizer
except:
    from config import DenConfig
    from den import DenForFineTuning
    from model import ChemBert
    from tokenizer import ChemBertTokenizer

from pathlib import Path, PosixPath
from transformers import BertTokenizer, AutoModel, BertConfig, BertModel

base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/DEN'

smi_encoder_path = base_path / 'models/CHEM-BERT/'
txt_encoder_path = base_path / 'models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
# encoder_type = 'smi'
encoder_type = 'all'


def load_den(path: PosixPath = model_path, encoder_type: str = encoder_type, ckpt_id: str = '') -> AutoModel:
    if ckpt_id == '':
        # ckpt_id = 'o6d6inqz'
        # ckpt_id = 'tfg33i7a'
        # ckpt_id = 't6xymgvr'
        # ckpt_id = 'nd9ckrb8'
        # ckpt_id = '6nn1f40f'
        # ckpt_id = 'l3x8us5x'
        # ckpt_id = 'mmaxprdp'
        # ckpt_id = '2h0kvj82'
        # ckpt_id = 'gjqy7ycv'
        # ckpt_id = 'wcv87cck'
        # ckpt_id = 'zs865hb4'
        # ckpt_id = 'kll2yqaq'
        # ckpt_id = 'h9n2m1qp'
        # ckpt_id = 'in4se4nz'
        ckpt_id = 'f9x97q2q'
        # ckpt_id = 'rk6ik8mf'
    print(f'load ckpt: {ckpt_id}')
    ckpt_all = torch.load(path / ckpt_id / 'pytorch_model.bin', map_location='cpu')

    # for key, value in ckpt_all.items():
    #     print(f'{key}: {value.shape}')

    if encoder_type == 'smi':
        ckpt = {}
        for key, value in ckpt_all.items():
            if key.startswith('model.den.smi_encoder.'):
                key = key[len('model.den.smi_encoder.'):]
                ckpt[key] = value
        
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        model = ChemBert(**config)

        model.load_state_dict(ckpt, strict=True)
    
    elif encoder_type == 'all':
        # build smi_encoder
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        smi_encoder = ChemBert(**config)
        smi_encoder.load_state_dict(torch.load(base_path / smi_encoder_path / 'pretrained_model.pt', map_location='cpu'), strict=True)

        # build model
        config = DenConfig()
        model = DenForFineTuning(smi_encoder=smi_encoder)

        # remove prefix model.
        for k in list(ckpt_all.keys()):
            if k.startswith('model.'):
                ckpt_all[k[len('model.'):]] = ckpt_all.pop(k)
        
        # remove prefix den.
        for k in list(ckpt_all.keys()):
            if k.startswith('den.'):
                ckpt_all[k[len('den.'):]] = ckpt_all.pop(k)

        # ignored keys
        ignored_keys = [
            "logit_scale", 
            "txt_projection.weight", 
            "smi_projection.weight", 
            "regressor.weight", 
            "regressor.bias", 
            "txt_cls.predictions.bias", 
            "txt_cls.predictions.transform.dense.weight", 
            "txt_cls.predictions.transform.dense.bias", 
            "txt_cls.predictions.transform.LayerNorm.weight", 
            "txt_cls.predictions.transform.LayerNorm.bias", 
            "txt_cls.predictions.decoder.weight", 
            "txt_cls.predictions.decoder.bias",
            "classifier.weight",
            "classifier.bias",
            'squeue',
            'squeue_ptr',
            "txt_projection.0.weight", 
            "txt_projection.0.bias", 
            "txt_projection.2.weight", 
            "txt_projection.2.bias", 
            "smi_projection.0.weight", 
            "smi_projection.0.bias", 
            "smi_projection.2.weight", 
            "smi_projection.2.bias",
            "smi_cls.predictions.bias", 
            "smi_cls.predictions.transform.dense.weight", 
            "smi_cls.predictions.transform.dense.bias", 
            "smi_cls.predictions.transform.LayerNorm.weight", 
            "smi_cls.predictions.transform.LayerNorm.bias", 
            "smi_cls.predictions.decoder.weight", 
            "smi_cls.predictions.decoder.bias",
            "txt2smi.weight",
            "txt2smi.bias"
        ]
        for k in list(ckpt_all.keys()):
            if k in ignored_keys:
                ckpt_all.pop(k)

        model.load_state_dict(ckpt_all, strict=True)
    
    return model

def load_den_tokenizer(path: PosixPath = model_path, encoder_type: str = encoder_type):
    if encoder_type == 'smi':
        return ChemBertTokenizer(vocab_path=base_path / smi_encoder_path / 'vocab.json')
    elif encoder_type == 'all':
        return ChemBertTokenizer(vocab_path=base_path / smi_encoder_path / 'vocab.json'), \
            BertTokenizer.from_pretrained(txt_encoder_path)



if __name__ == '__main__':
    # model = load_den(ckpt_id='rmjjrxuc')
    # model = load_den(ckpt_id='y9z93dma')
    # model = load_den(ckpt_id='uzqk4lp6')
    # model = load_den(ckpt_id='si2gtn3d')
    model = load_den(ckpt_id='0chcylaj')
    # tokenizer = load_den_tokenizer()
    print(model)
    # print(tokenizer)