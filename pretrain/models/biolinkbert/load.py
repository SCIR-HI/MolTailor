import sys
import torch

from pathlib import Path, PosixPath
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/BioLinkBERT-base'

def load_biolinkbert(path: PosixPath = model_path, ckpt_id: str = '') -> AutoModel:
    config = BertConfig().from_pretrained(path)

    if ckpt_id:
        pass
        ckpt = torch.load(path / 'ckpt' / ckpt_id / 'last.pt')
    else:
        ckpt = torch.load(path / 'pytorch_model.bin')

    # show param names
    for k, v in ckpt.items():
        print(f'{k}: {v.shape}')
    
    # remove 'bert.' prefix in ckpt keys
    for k in list(ckpt.keys()):
        if k.startswith('bert.'):
            ckpt[k[len('bert.'):]] = ckpt.pop(k)
        if k.startswith('encoder.'):
            ckpt[k[len('encoder.'):]] = ckpt.pop(k)
            
    # remove ignored keys
    ignored_keys = [
        "cls.predictions.bias", 
        "cls.predictions.transform.dense.weight", 
        "cls.predictions.transform.dense.bias", 
        "cls.predictions.transform.LayerNorm.weight", 
        "cls.predictions.transform.LayerNorm.bias", 
        "cls.predictions.decoder.weight", 
        "cls.predictions.decoder.bias",
        "cls.seq_relationship.weight",
        "cls.seq_relationship.bias",
        "regressor.weight", 
        "regressor.bias",
        "pooler.dense.weight",
        "pooler.dense.bias",
        "embeddings.position_ids"
    ]
    
    for k in list(ckpt.keys()):
        if any([k.endswith(x) for x in ignored_keys]):
            ckpt.pop(k)

    model = BertModel(config, add_pooling_layer=False)
    model.load_state_dict(ckpt)
    
    return model

def load_biolinkbert_tokenizer(path: PosixPath = model_path) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(path)


if __name__ == '__main__':
    model = load_biolinkbert(ckpt_id='lpbpxt7w')
    print(model)