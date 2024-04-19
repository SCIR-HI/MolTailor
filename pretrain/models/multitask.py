import copy
from typing import Any, Optional, Tuple, Union, Dict
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
from torch import nn
import torch.nn as nn

import sys
from pathlib import Path, PosixPath

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

try:
    from config import DenConfig
    from load import load_model
    from den import DenModel, DenPreTrainedModel
except:
    from .config import DenConfig
    from .load import load_model
    from .den import DenModel, DenPreTrainedModel

import argparse
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, BertForMaskedLM, BertModel, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MaskedLMOutput, ModelOutput, dataclass
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW


config = DenConfig()
config.ckpt_path = str(base_dir.parent / config.ckpt_path)

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
class DenForMultiTaskPreTrainingOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    loss_mlm_smi: Optional[torch.FloatTensor] = None
    loss_cl: Optional[torch.FloatTensor] = None
    loss_mtr: Optional[torch.FloatTensor] = None


class DenPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class DenLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = DenPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class DenOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = DenLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class DenForMultiTaskPreTraining(DenPreTrainedModel):
    def __init__(self, config: DenConfig, smi_encoder, txt_cls = None, smi_cls = None):
        super().__init__(config)
        self.txt_config = config
        self.smi_config = smi_encoder.config


        # mlm
        # if txt_cls is None:
        #     txt_cls = DenOnlyMLMHead(config)

        # config for DenOnlyMLMHead
        # hidden_size, vocab_size, hidden_act, layer_norm_eps
        # if smi_cls is None:
        #     config_smi_cls = copy.deepcopy(config)
        #     config_smi_cls.hidden_size = self.smi_config.feature_dim
        #     config_smi_cls.vocab_size = self.smi_config.vocab_size
        #     self.smi_cls = DenOnlyMLMHead(config_smi_cls)

        # cl
        # self.txt_hidden_size = config.hidden_size
        # self.smi_hidden_size = smi_encoder.config.feature_dim
        # self.projection_dim = config.projection_dim
        
        # self.txt_projection = nn.Linear(self.txt_hidden_size, self.projection_dim, bias=False)
        # self.smi_projection = nn.Linear(self.smi_hidden_size, self.projection_dim, bias=False)

        # self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))

        # mtr
        self.regressor = nn.Linear(config.hidden_size, config.num_labels)
        self.criterion_mtr = nn.MSELoss(reduction='none')

        # init
        self.apply(self._initialize_weights)

        self.den = DenModel(config, smi_encoder)
        # mlm
        # if txt_cls is not None:
        #     self.txt_cls = txt_cls
        # self.txt_cls = None

        # self.tie_weights()

        # freeze smi_encoder
        # for param in self.den.smi_encoder.parameters():
        #     param.requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, 'den', self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError
        
    def get_output_embeddings(self):
        txt_out_emb = self.txt_cls.predictions.decoder if self.txt_cls is not None else None
        smi_out_emb = self.smi_cls.predictions.decoder if self.smi_cls is not None else None
        return txt_out_emb, smi_out_emb
    
    def set_output_embeddings(self, new_embeddings):
        raise NotImplementedError('Error: set_output_embeddings is not expected to be called in DenForMaskedLM')

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        if getattr(self.config, "tie_word_embeddings", True):
            txt_out_emb, smi_out_emb = self.get_output_embeddings()
            txt_in_emb, smi_in_emb = self.get_input_embeddings()

            if txt_out_emb is not None and txt_in_emb is not None:
                self._tie_or_clone_weights(txt_out_emb, txt_in_emb)
            if smi_out_emb is not None and smi_in_emb is not None:
                self._tie_or_clone_weights(smi_out_emb, smi_in_emb)

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                raise NotImplementedError('Error: _tie_weights is not expected to be called in DenForMaskedLM')
                # module._tie_weights()

    def forward(self,
                txt,
                smi,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,)-> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.den(txt, smi, output_attentions, output_hidden_states, return_dict)

        # sequnce_output_smi = outputs.last_hidden_state_smi
        # sequnce_output_smi_mlm = outputs.last_hidden_state_smi_mlm
        # sequnce_output_txt_mlm = outputs.last_hidden_state_txt_mlm
        # sequnce_output_txt_cl = outputs.last_hidden_state_txt_cl
        sequnce_output_txt_mtr = outputs.last_hidden_state_txt_mtr

        # mlm
        # loss_fct = CrossEntropyLoss()
        # # prediction_scores_txt = self.txt_cls(sequnce_output_txt_mlm)
        # # loss_mlm = loss_fct(prediction_scores_txt.view(-1, self.config.vocab_size), txt['labels-mlm'].view(-1))

        # prediction_scores_smi = self.smi_cls(sequnce_output_smi_mlm)
        # loss_mlm_smi = loss_fct(prediction_scores_smi.view(-1, self.smi_config.vocab_size), smi['labels-mlm'].view(-1))
        
        # cl
        # txt_embeds = sequnce_output_txt_cl[:, 0, :]
        # txt_embeds = self.txt_projection(txt_embeds)
        # smi_embeds = sequnce_output_smi[:, 0, :]
        # smi_embeds = self.smi_projection(smi_embeds)
        # # normalize features
        # txt_embeds = txt_embeds / txt_embeds.norm(p=2, dim=-1, keepdim=True)
        # smi_embeds = smi_embeds / smi_embeds.norm(p=2, dim=-1, keepdim=True)
        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_txt = torch.matmul(txt_embeds, smi_embeds.t()) * logit_scale
        # # calculate loss
        # loss_cl = clip_loss(logits_per_txt)

        # mtr
        # logits_mtr = self.regressor(sequnce_output_txt_mtr[:, 0, :]) # cls
        logits_mtr = self.regressor(sequnce_output_txt_mtr.mean(dim=1)) # mean
        loss_mtr = self.criterion_mtr(logits_mtr, txt['labels-mtr'])
        loss_mtr_masked = loss_mtr * txt['labels-mtr-mask']
        loss_mtr = loss_mtr_masked.sum() / txt['labels-mtr-mask'].sum()


        # loss = loss_mlm + loss_cl + loss_mtr
        # loss = loss_mtr + loss_mlm_smi
        loss = loss_mtr

        if not return_dict:
            # return (loss, loss_mtr, loss_mlm_smi)
            return (loss, loss_mtr)
        
        return DenForMultiTaskPreTrainingOutput(
            loss=loss,
            loss_mtr=loss_mtr,
            # loss_mlm_smi=loss_mlm_smi,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        raise NotImplementedError('Error: prepare_inputs_for_generation is not expected to be called in DenForMaskedLM')
    

class DenForPreTraining(pl.LightningModule):
    def __init__(self, 
                 config: DenConfig = config,
                 lr: float = 5e-5,
                 num_training_steps: int = 100000,
                 num_warmup_steps: int = 10000,) -> None:
        super().__init__()
        model_name_txt = config.model_name_txt
        model_name_smi = config.model_name_smi

        txt_model = load_model(model_name_txt)
        txt_cls = txt_model.cls
        smi_encoder = load_model(model_name_smi)

        self.save_hyperparameters()
        self.model = DenForMultiTaskPreTraining(config, smi_encoder, txt_cls)

    def forward(self, txt, smi):
        return self.model(txt, smi)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        txt = batch['txt']
        smi = batch['smi']

        outputs = self(txt, smi)
        loss = outputs[0]

        self.log('train/loss', loss, prog_bar=True)
        # self.log('train/loss-mlm-smi', outputs.loss_mlm_smi)
        # self.log('train/loss-cl', outputs.loss_cl)
        self.log('train/loss-mtr', outputs.loss_mtr)

        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        txt = batch['txt']
        smi = batch['smi']

        outputs = self(txt, smi)
        loss = outputs[0]

        self.log('val/loss', loss, sync_dist=True)
        # self.log('val/loss-mlm-smi', outputs.loss_mlm_smi, sync_dist=True)
        # self.log('val/loss-cl', outputs.loss_cl, sync_dist=True)
        self.log('val/loss-mtr', outputs.loss_mtr, sync_dist=True)

        return loss
    
    def configure_optimizers(self) -> Any:
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)

        scheduler = {
            'scheduler': get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.hparams.num_warmup_steps, 
                num_training_steps=self.hparams.num_training_steps
            ),
            'interval': 'step',
            'frequency': 1
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}




if __name__ == '__main__':
    # convert ckpt file
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_id', type=str, default='f9x97q2q')
    args = parser.parse_args()
    ckpt_id = args.ckpt_id

    ckpt_path = base_dir.parent / f'models/DEN/{ckpt_id}/last.ckpt'
    save_path = base_dir.parent / f'models/DEN/{ckpt_id}/pytorch_model.bin'

    print(f'ckpt_path: {ckpt_path}') 
    print(f'save_path: {save_path}')

    model = DenForPreTraining.load_from_checkpoint(ckpt_path, map_location='cpu')

    # save state_dict
    state_dict = model.state_dict()
    torch.save(state_dict, save_path)
    print(f'save state_dict done!: {ckpt_id}')

    


            

        
            