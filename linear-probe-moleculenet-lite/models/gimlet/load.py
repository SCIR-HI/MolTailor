import sys
import torch

from pathlib import Path, PosixPath
from transformers import AutoTokenizer, T5ForConditionalGeneration


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/gimlet'


def load_gimlet(path: PosixPath = model_path) -> T5ForConditionalGeneration:
    model = T5ForConditionalGeneration.from_pretrained(path)
    return model

def load_gimlet_tokenizer(path: PosixPath = model_path) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path, model_max_length=512)
    return tokenizer


if __name__ == '__main__':
    smiles = 'CCC(O)C(=O)O'
    tokenizer = load_gimlet_tokenizer()
    model = load_gimlet()

    inputs = tokenizer(smiles, return_tensors='pt')
    encoder_outputs = model.encoder(**inputs)

    encoder_outputs.last_hidden_state.mean(dim=1).shape
