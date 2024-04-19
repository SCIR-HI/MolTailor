import torch
from torch import nn
import torch.nn as nn

import sys
from pathlib import Path, PosixPath

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

try:
    from bert import BertModel
    from config import DenConfig
    from load import load_model
    from chembert.load import ChemBert
except:
    from .bert import BertModel
    from .config import DenConfig
    from .load import load_model
    from .chembert.load import ChemBert

from typing import Optional, Tuple

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput, dataclass

config = DenConfig()
config.ckpt_path = str(base_dir.parent / config.ckpt_path)


@dataclass
class DenModelOutput(ModelOutput):

    # last_hidden_state_txt = sequence_output_txt,
    # last_hidden_state_smi = sequence_output_smi,
    # hidden_states_txt = outputs.hidden_states_txt,
    # hidden_states_smi = outputs.hidden_states_smi,
    # attentions = outputs.attentions,
        
    last_hidden_state_smi: torch.FloatTensor = None
    last_hidden_state_smi_mlm: torch.FloatTensor = None
    # last_hidden_state_txt_mlm: torch.FloatTensor = None
    # last_hidden_state_txt_cl: torch.FloatTensor = None
    last_hidden_state_txt_mtr: torch.FloatTensor = None

@dataclass
class DenForFineTuningOutput(ModelOutput):

    last_hidden_state_smi: torch.FloatTensor = None
    last_hidden_state_txt: torch.FloatTensor = None

class DenPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DenConfig
    # base_model_prefix = "bert"
    # supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, BertEncoder):
    #         module.gradient_checkpointing = value




class DenModel(DenPreTrainedModel):
    def __init__(self, config: DenConfig, smi_encoder: ChemBert):
        super().__init__(config)
        self.config = config
        self.smi_config = smi_encoder.config

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        
        self.attention_head_size =  int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.smi_key = nn.Linear(self.smi_config.feature_dim, self.all_head_size)
        self.smi_value = nn.Linear(self.smi_config.feature_dim, self.all_head_size)


        # Initialize weights and apply final processing
        self.post_init()

        self.txt_encoder = BertModel(config, add_pooling_layer=False)
        self.load_txt_encoder(config.ckpt_path)
        self.smi_encoder = smi_encoder
        
    def load_txt_encoder(self, path):
        ckpt = torch.load(path)

        # remove 'bert.' prefix in ckpt keys
        for k in list(ckpt.keys()):
            if k.startswith('bert.'):
                ckpt[k[len('bert.'):]] = ckpt.pop(k)
                
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
            "pooler.dense.weight",
            "pooler.dense.bias",
        ]
        
        for k in list(ckpt.keys()):
            if any([k.endswith(x) for x in ignored_keys]):
                ckpt.pop(k)
        
        self.txt_encoder.load_state_dict(ckpt, strict=True)


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def get_input_embeddings(self):
        return self.txt_encoder.embeddings.word_embeddings, self.smi_encoder.embedding.token

    def set_input_embeddings(self, value: nn.Module):
        raise NotImplementedError('Error: set_input_embeddings is not expected to be called in DenModel')
    
    def _prune_heads(self, heads_to_prune: dict):
        raise NotImplementedError('Error: _prune_heads is not expected to be called in DenModel')
    

    def forward(self,
                txt,
                smi,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )


        outputs_smi = self.smi_encoder(
            input = smi['input'],
            imask = smi['imask'],
            amask = smi['amask'],
            amatx = smi['amatx'],
        )

        # outputs_smi_mlm = self.smi_encoder(
        #     input = smi['input-mlm'],
        #     imask = smi['imask'],
        #     amask = smi['amask'],
        #     amatx = smi['amatx'],
        # )

        # outputs_txt_mlm = self.txt_encoder(
        #     input_ids = txt['input-mlm'],
        #     attention_mask = txt['imask'],
        #     output_attentions = output_attentions,
        #     output_hidden_states = output_hidden_states,
        #     return_dict = return_dict,
        # )

        # outputs_txt_cl = self.txt_encoder(
        #     input_ids = txt['input-cl'],
        #     attention_mask = txt['imask'],
        #     output_attentions = output_attentions,
        #     output_hidden_states = output_hidden_states,
        #     return_dict = return_dict,
        # )

        smi_key_layer = self.transpose_for_scores(self.smi_key(outputs_smi))
        smi_value_layer = self.transpose_for_scores(self.smi_value(outputs_smi))

        # assert smi_key_layer.shape == (outputs_smi.shape[0], self.num_attention_heads, outputs_smi.shape[1], self.attention_head_size)

        past_key_value = torch.stack((smi_key_layer, smi_value_layer), dim=0)
        
        extended_attention_mask_txt = txt['imask-mtr'][:, None, None, :]
        extended_attention_mask_smi = (~smi['imask'][:, None, None, :]).to(dtype=extended_attention_mask_txt.dtype)
        extended_attention_mask = torch.cat([extended_attention_mask_smi, extended_attention_mask_txt], dim=-1)

        past_key_values = {
            'past_key_value': past_key_value,
            'attention_mask': extended_attention_mask,
        }
        
        outputs_txt_mtr = self.txt_encoder(
            input_ids = txt['input-mtr'],
            attention_mask = txt['imask-mtr'],
            past_key_values = past_key_values,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
        )

        sequence_output_smi = outputs_smi
        # sequence_output_smi_mlm = outputs_smi_mlm
        # sequence_output_txt_mlm = outputs_txt_mlm.last_hidden_state
        # sequence_output_txt_cl = outputs_txt_cl.last_hidden_state
        sequence_output_txt_mtr = outputs_txt_mtr.last_hidden_state

        if not return_dict:
            # return (sequence_output_smi, sequence_output_smi_mlm, sequence_output_txt_mtr)
            return (sequence_output_smi, sequence_output_txt_mtr)
        
        return DenModelOutput(
            last_hidden_state_smi = sequence_output_smi,
            # last_hidden_state_smi_mlm = sequence_output_smi_mlm,
            # last_hidden_state_txt_mlm = sequence_output_txt_mlm,
            last_hidden_state_txt_mtr = sequence_output_txt_mtr,
        )


class DenForFineTuning(DenPreTrainedModel):
    def __init__(self, config: DenConfig = config, smi_encoder = None):
        super().__init__(config)
        self.config = config
        self.smi_config = smi_encoder.config

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        
        self.attention_head_size =  int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.smi_key = nn.Linear(self.smi_config.feature_dim, self.all_head_size)
        self.smi_value = nn.Linear(self.smi_config.feature_dim, self.all_head_size)


        # Initialize weights and apply final processing
        self.apply(self._initialize_weights)

        self.txt_encoder = BertModel(config, add_pooling_layer=False)
        self.load_txt_encoder(config.ckpt_path)
        self.smi_encoder = smi_encoder

    def load_txt_encoder(self, path):
        ckpt = torch.load(path)

        # remove 'bert.' prefix in ckpt keys
        for k in list(ckpt.keys()):
            if k.startswith('bert.'):
                ckpt[k[len('bert.'):]] = ckpt.pop(k)
                
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
            "pooler.dense.weight",
            "pooler.dense.bias",
        ]
        
        for k in list(ckpt.keys()):
            if any([k.endswith(x) for x in ignored_keys]):
                ckpt.pop(k)
        
        self.txt_encoder.load_state_dict(ckpt, strict=True)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                txt,
                smi,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs_smi = self.smi_encoder(
            input = smi['input'],
            imask = smi['imask'],
            amask = smi['amask'],
            amatx = smi['amatx'],
        )


        smi_key_layer = self.transpose_for_scores(self.smi_key(outputs_smi))
        smi_value_layer = self.transpose_for_scores(self.smi_value(outputs_smi))

        # assert smi_key_layer.shape == (outputs_smi.shape[0], self.num_attention_heads, outputs_smi.shape[1], self.attention_head_size)

        past_key_value = torch.stack((smi_key_layer, smi_value_layer), dim=0)
        
        extended_attention_mask_txt = txt['imask'][:, None, None, :]
        extended_attention_mask_smi = (~smi['imask'][:, None, None, :]).to(dtype=extended_attention_mask_txt.dtype)
        extended_attention_mask = torch.cat([extended_attention_mask_smi, extended_attention_mask_txt], dim=-1)

        past_key_values = {
            'past_key_value': past_key_value,
            'attention_mask': extended_attention_mask,
        }
        
        outputs_txt = self.txt_encoder(
            input_ids = txt['input'],
            attention_mask = txt['imask'],
            past_key_values = past_key_values,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
        )

        sequence_output_smi = outputs_smi
        sequence_output_txt = outputs_txt.last_hidden_state

        if not return_dict:
            return (sequence_output_smi, sequence_output_txt)
        

        return DenForFineTuningOutput(
            last_hidden_state_smi = sequence_output_smi,
            last_hidden_state_txt = sequence_output_txt,
        )
        


if __name__ == '__main__':
    config = DenConfig()
    smi_encoder = load_model('CHEM-BERT')

    config.ckpt_path = base_dir.parent / 'models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/pytorch_model.bin'

    model = DenModel(config, smi_encoder)

    # construct fake data
    txt = {
        'input-mlm': torch.randint(0, 100, (2, 128), dtype=torch.long),
        'input-cl': torch.randint(0, 100, (2, 128), dtype=torch.long),
        'imask': torch.randint(0, 2, (2, 128), dtype=torch.float),
        'input-mtr': torch.randint(0, 100, (2, 128), dtype=torch.long),
        'imask-mtr': torch.randint(0, 2, (2, 128), dtype=torch.float),
    }

    smi = {
        'input': torch.randint(0, 47, (2, 128), dtype=torch.long),
        'imask': torch.randint(0, 2, (2, 128), dtype=torch.bool),
        'amask': torch.randint(0, 2, (2, 128), dtype=torch.float),
        'amatx': torch.randint(0, 2, (2, 256, 256), dtype=torch.float),
    }


    with torch.no_grad():
        outputs = model(txt, smi, return_dict=True)
        # print(outputs)

    for name, param in model.named_parameters():
        print(name, param.shape)



