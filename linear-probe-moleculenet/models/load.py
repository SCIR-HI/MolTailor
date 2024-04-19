'''
Author: simwit 517992857@qq.com
Date: 2023-07-21 15:21:40
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-12 15:16:50
FilePath: /workspace/01-st/finetune-moleculenet/models/load.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from models.chemberta_77m_mtr.load import load_chemberta_77m_mtr, load_chemberta_77m_mtr_tokenizer
from models.chemberta_10m_mtr.load import load_chemberta_10m_mtr, load_chemberta_10m_mtr_tokenizer
from models.pubmedbert.load import load_pubmedbert, load_pubmedbert_tokenizer
from models.scibert.load import load_scibert, load_scibert_tokenizer
from models.chemberta_77m_mlm.load import load_chemberta_77m_mlm, load_chemberta_77m_mlm_tokenizer
from models.den_s.load import load_den_s, load_den_s_tokenizer
from models.chembert.load import load_chembert, load_chembert_tokenizer
from models.den_cl.load import load_den_cl, load_den_cl_tokenizer
from models.den_cross.load import load_den_cross, load_den_cross_tokenizer
from models.bert.load import load_bert, load_bert_tokenizer
from models.roberta.load import load_roberta, load_roberta_tokenizer
from models.uni_mol.load import load_uni_mol
from models.den.load import load_den, load_den_tokenizer
from models.kv_plm.load import load_kv_plm, load_kv_plm_tokenizer
from models.molt5.load import load_molt5, load_molt5_tokenizer
from models.t5.load import load_t5, load_t5_tokenizer
from models.kcl.load import load_kcl, load_kcl_tokenizer
from models.grover.load import load_grover, load_grover_tokenizer
from models.molclr.load import load_molclr, load_molclr_tokenizer
from models.clamp.load import load_clamp
from models.den_chemberta.load import load_den_chemberta, load_den_chemberta_tokenizer
from models.molebert.load import load_molebert, load_molebert_tokenizer
from models.gimlet.load import load_gimlet, load_gimlet_tokenizer
from models.tct5.load import load_tct5, load_tct5_tokenizer
from models.momu.load import load_momu, load_momu_tokenizer
from models.biolinkbert.load import load_biolinkbert, load_biolinkbert_tokenizer
from models.momu_te.load import load_momu_te, load_momu_te_tokenizer

def load_den_smi(encoder_type: str = 'smi'):
    return load_den(encoder_type=encoder_type)

def load_den_all(encoder_type: str = 'all', ckpt_id: str = ''):
    return load_den(encoder_type=encoder_type, ckpt_id = ckpt_id)

def load_den_smi_tokenizer(encoder_type: str = 'smi'):
    return load_den_tokenizer(encoder_type=encoder_type)

def load_den_all_tokenizer(encoder_type: str = 'all'):
    return load_den_tokenizer(encoder_type=encoder_type)

def load_grover_large(type='large'):
    return load_grover(type=type)

def load_grover_base(type='base'):
    return load_grover(type=type)

name2model = {
    'ChemBERTa-77M-MTR': load_chemberta_77m_mtr,
    'ChemBERTa-10M-MTR': load_chemberta_10m_mtr,
    'PubMedBERT': load_pubmedbert,
    'SciBERT': load_scibert,
    'ChemBERTa-77M-MLM': load_chemberta_77m_mlm,
    'DEN-S': load_den_s,
    'CHEM-BERT': load_chembert,
    'DEN-CL-SMI': load_den_cl,
    'DEN-CL-TXT': load_den_cl,
    'DEN-Cross': load_den_cross,
    'DEN-Cross-SMI': load_den_cross,
    'DEN-Cross-TXT': load_den_cross,
    'BERT': load_bert,
    'RoBERTa': load_roberta,
    'Uni-Mol': load_uni_mol,
    'DEN-SMI': load_den_smi,
    'DEN': load_den_all,
    'KV-PLM': load_kv_plm,
    'MolT5': load_molt5,
    'T5': load_t5,
    'KCL': load_kcl,
    'Grover': load_grover_large,
    'Grover-Base': load_grover_base,
    'MolCLR': load_molclr,
    'CLAMP': load_clamp,
    'DEN-ChemBERTa': load_den_chemberta,
    'Mole-BERT': load_molebert,
    'GIMLET': load_gimlet,
    'TCT5': load_tct5,
    'MoMu': load_momu,
    'BioLinkBERT': load_biolinkbert,
    'MoMu-TE': load_momu_te,
}

name2tokenizer = {
    'ChemBERTa-77M-MTR': load_chemberta_77m_mtr_tokenizer,
    'ChemBERTa-10M-MTR': load_chemberta_10m_mtr_tokenizer,
    'PubMedBERT': load_pubmedbert_tokenizer,
    'SciBERT': load_scibert_tokenizer,
    'ChemBERTa-77M-MLM': load_chemberta_77m_mlm_tokenizer,
    'DEN-S': load_den_s_tokenizer,
    'CHEM-BERT': load_chembert_tokenizer,
    'DEN-CL-SMI': load_den_cl_tokenizer,
    'DEN-CL-TXT': load_den_cl_tokenizer,
    'DEN-Cross': load_den_cross_tokenizer,
    'DEN-Cross-SMI': load_den_cross_tokenizer,
    'DEN-Cross-TXT': load_den_cross_tokenizer,
    'BERT': load_bert_tokenizer,
    'RoBERTa': load_roberta_tokenizer,
    'DEN-SMI': load_den_smi_tokenizer,
    'DEN': load_den_all_tokenizer,
    'KV-PLM': load_kv_plm_tokenizer,
    'MolT5': load_molt5_tokenizer,
    'T5': load_t5_tokenizer,
    'KCL': load_kcl_tokenizer,
    'Grover': load_grover_tokenizer,
    'Grover-Base': load_grover_tokenizer, # 'Grover-Base' and 'Grover' share the same tokenizer
    'MolCLR': load_molclr_tokenizer,
    'DEN-ChemBERTa': load_den_chemberta_tokenizer,
    'Mole-BERT': load_molebert_tokenizer,
    'GIMLET': load_gimlet_tokenizer,
    'TCT5': load_tct5_tokenizer,
    'MoMu': load_momu_tokenizer,
    'BioLinkBERT': load_biolinkbert_tokenizer,
    'MoMu-TE': load_momu_te_tokenizer,
}


def load_model(model_name: str, *args, **kwargs):
    return name2model[model_name](*args, **kwargs)

def load_tokenizer(model_name: str, *args, **kwargs):
    return name2tokenizer[model_name](*args, **kwargs)

