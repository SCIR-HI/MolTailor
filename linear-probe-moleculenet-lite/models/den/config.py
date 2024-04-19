'''
Author: simwit 517992857@qq.com
Date: 2023-07-31 09:51:21
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-06 17:41:12
FilePath: /workspace/01-st/pretrain-dual/models/config.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
from transformers import BertConfig

"""
    config:
        cross_hidden_size, 
        cross_num_attention_heads,

        txt_vocab_size,
        txt_hidden_size,
        txt_attention_probs_dropout_prob,
        txt_layer_norm_eps,
        txt_hidden_dropout_prob,
        txt_intermediate_size,
        txt_hidden_act,

        smi_vocab_size,
        smi_hidden_size,
        smi_attention_probs_dropout_prob,
        smi_layer_norm_eps,
        smi_hidden_dropout_prob,
        smi_intermediate_size,
        smi_hidden_act,


        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        max_position_embeddings,
        layer_norm_eps,
        hidden_dropout_prob,
        intermediate_size,
        hidden_act,
"""

class DenConfig(BertConfig):
    def __init__(self, 
                 vocab_size=30522, 
                 hidden_size=768, 
                 num_hidden_layers=12, 
                 num_attention_heads=12, 
                 intermediate_size=3072, 
                 hidden_act="gelu", 
                 hidden_dropout_prob=0.1, 
                 attention_probs_dropout_prob=0.1, 
                 max_position_embeddings=512, 
                 type_vocab_size=2, 
                 initializer_range=0.02, 
                 layer_norm_eps=1e-12, 
                 pad_token_id=0, 
                 position_embedding_type="absolute", 
                 use_cache=True, 
                 classifier_dropout=None,
                 ckpt_path='models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/pytorch_model.bin',
                 projection_dim=512,
                 logit_scale_init_value=2.6592,
                 num_labels=209,
                 model_name_txt='PubMedBERT',
                 model_name_smi='CHEM-BERT',
                 **kwargs):
        super().__init__(
            vocab_size, 
            hidden_size, 
            num_hidden_layers, 
            num_attention_heads, 
            intermediate_size, 
            hidden_act, 
            hidden_dropout_prob, 
            attention_probs_dropout_prob, 
            max_position_embeddings, 
            type_vocab_size, 
            initializer_range, 
            layer_norm_eps, 
            pad_token_id, 
            position_embedding_type, 
            use_cache, 
            classifier_dropout, 
            **kwargs
        )

        self.ckpt_path = ckpt_path
        self.model_name_txt = model_name_txt
        self.model_name_smi = model_name_smi

        # mtr
        self.cross_layer_num = 3
        self.cross_layer_idx = [self.num_hidden_layers - 1 - i for i in range(self.cross_layer_num)]

        # cl
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

        # mtr
        self.num_labels = num_labels




