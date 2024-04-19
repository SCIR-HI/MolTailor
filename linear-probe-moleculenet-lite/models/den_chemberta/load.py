'''
Author: simwit 517992857@qq.com
Date: 2023-08-07 09:47:52
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-12 18:36:39
FilePath: /workspace/01-st/finetune-moleculenet/models/den_chemberta/load.py
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
from transformers import BertTokenizer, AutoModel, BertConfig, BertModel, RobertaTokenizer, RobertaModel, RobertaConfig

base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/DEN'

smi_encoder_path = base_path / 'models/ChemBERTa-10M-MTR/'
txt_encoder_path = base_path / 'models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
# encoder_type = 'smi'
encoder_type = 'all'


def load_chemberta_10m_mtr(path: PosixPath = model_path) -> AutoModel:
    config = RobertaConfig().from_pretrained(path)
    ckpt = torch.load(path / 'pytorch_model.bin')
    
    # remove 'roberta.' prefix in ckpt keys
    for k in list(ckpt.keys()):
        if k.startswith('roberta.'):
            ckpt[k[len('roberta.'):]] = ckpt.pop(k)
            
    # remove ignored keys
    ignored_keys = [
        "embeddings.position_ids", 
        "norm_mean", 
        "norm_std", 
        "regression.dense.weight", 
        "regression.dense.bias", 
        "regression.out_proj.weight", 
        "regression.out_proj.bias"
    ]
    
    for k in list(ckpt.keys()):
        if any([k.endswith(x) for x in ignored_keys]):
            ckpt.pop(k)
    model = RobertaModel(config)
    model.pooler = None
    model.load_state_dict(ckpt)
    
    return model


def load_den_chemberta(path: PosixPath = model_path, encoder_type: str = encoder_type, ckpt_id: str = '') -> AutoModel:
    if ckpt_id == '':
        ckpt_id = 'xqvnjohh'
    print(f'ChemBERTa load ckpt_id: {ckpt_id}')
    
    ckpt_all = torch.load(path / ckpt_id / 'pytorch_model.bin', map_location='cpu')

    # for key, value in ckpt_all.items():
    #     print(f'{key}: {value.shape}')

    if encoder_type == 'smi':
        ckpt = {}
        for key, value in ckpt_all.items():
            if key.startswith('model.den.smi_encoder.'):
                key = key[len('model.den.smi_encoder.'):]
                ckpt[key] = value
        
        model = load_chemberta_10m_mtr(smi_encoder_path)
        model.load_state_dict(ckpt, strict=True)
    
    elif encoder_type == 'all':
        # build smi_encoder
        smi_encoder = load_chemberta_10m_mtr(smi_encoder_path)

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
            "squeue", 
            "squeue_ptr", 
            "txt_projection.0.weight", 
            "txt_projection.0.bias", 
            "txt_projection.2.weight", 
            "txt_projection.2.bias"
        ]
        for k in list(ckpt_all.keys()):
            if k in ignored_keys:
                ckpt_all.pop(k)

        model.load_state_dict(ckpt_all, strict=True)
    
    return model

def load_den_chemberta_tokenizer(path: PosixPath = model_path, encoder_type: str = encoder_type):
    if encoder_type == 'smi':
        return RobertaTokenizer.from_pretrained(smi_encoder_path)
    elif encoder_type == 'all':
        return RobertaTokenizer.from_pretrained(smi_encoder_path), \
            BertTokenizer.from_pretrained(txt_encoder_path)



if __name__ == '__main__':
    model = load_den_chemberta()
    tokenizer = load_den_chemberta_tokenizer()
    print(model)
    print(tokenizer)