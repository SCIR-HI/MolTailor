import sys
import torch

from pathlib import Path, PosixPath
from transformers import BertConfig, BertForMaskedLM, BertTokenizer, BertModel


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/bert-base-uncased'

def load_bert(path: PosixPath = model_path) -> BertModel:
    config = BertConfig.from_pretrained(path)
    ckpt = torch.load(path / 'pytorch_model.bin')

    # for name, param in ckpt.items():
    #     print(f'{name}: {param.shape}')

    # remove ignored keys
    ignored_keys = [
        "bert.pooler.dense.weight", 
        "bert.pooler.dense.bias",
        "cls.seq_relationship.weight",
        "cls.seq_relationship.bias",
    ]
    
    for k in list(ckpt.keys()):
        if any([k.endswith(x) for x in ignored_keys]):
            ckpt.pop(k)

    # change LayerNorm.gamma to LayerNorm.weight, LayerNorm.beta to LayerNorm.bias
    for k in list(ckpt.keys()):
        if k.endswith('LayerNorm.gamma'):
            ckpt[k.replace('gamma', 'weight')] = ckpt.pop(k)
        elif k.endswith('LayerNorm.beta'):
            ckpt[k.replace('beta', 'bias')] = ckpt.pop(k)

    # set cls.predictions.decoder.bias = cls.predictions.bias
    ckpt['cls.predictions.decoder.bias'] = ckpt['cls.predictions.bias']

    model = BertForMaskedLM(config)
    model.load_state_dict(ckpt)
    
    return model

def load_bert_tokenizer(path: PosixPath = model_path) -> BertTokenizer:
    return BertTokenizer.from_pretrained(path)


if __name__ == '__main__':
    model = load_bert()
    print(model)
    print(load_bert_tokenizer())

    print('start')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name)
    print('end')
