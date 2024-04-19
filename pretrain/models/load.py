import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from models.pubmedbert.load import load_pubmedbert, load_pubmedbert_tokenizer
from models.scibert.load import load_scibert, load_scibert_tokenizer
from models.chembert.load import load_chembert, load_chembert_tokenizer
from models.bert_uncased.load import load_bert, load_bert_tokenizer
from models.biolinkbert.load import load_biolinkbert, load_biolinkbert_tokenizer

name2model = {
    'PubMedBERT': load_pubmedbert,
    'SciBERT': load_scibert,
    'CHEM-BERT': load_chembert,
    'BERT': load_bert,
    'BioLinkBERT': load_biolinkbert,
}

name2tokenizer = {
    'PubMedBERT': load_pubmedbert_tokenizer,
    'SciBERT': load_scibert_tokenizer,
    'CHEM-BERT': load_chembert_tokenizer,
    'BERT': load_bert_tokenizer,
    'BioLinkBERT': load_biolinkbert_tokenizer,
}


def load_model(model_name: str):
    return name2model[model_name]()

def load_tokenizer(model_name: str):
    return name2tokenizer[model_name]()

