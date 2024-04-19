import sys
from pathlib import Path, PosixPath
from typing import Union, Any, Tuple, Dict, List, Mapping, Optional

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
data_dir = base_dir / 'data'

import torch
from dataclasses import dataclass
from transformers import DataCollatorForLanguageModeling, BertTokenizer

from models.chembert.tokenizer import ChemBertTokenizer

@dataclass
class CustomDataCollatorForMTP(DataCollatorForLanguageModeling):
    txt_tokenizer: BertTokenizer = None
    smi_tokenizer: ChemBertTokenizer = None

    def __post_init__(self):
        if self.mlm and self.txt_tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.

        txt_mtr_tokenized = self.txt_tokenizer.batch_encode_plus(
            [
                example['task_description']
                for example in examples
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        batch_txt = {
            # 'input-cl': batch_txt_cl['input_ids'],
            # 'imask': batch_txt_cl['attention_mask'],
            # 'special_tokens_mask': batch_txt_cl['special_tokens_mask'],
            'input-mtr': txt_mtr_tokenized['input_ids'],
            'imask-mtr': txt_mtr_tokenized['attention_mask'],
            'labels-mtr': torch.cat([example['labels-mtr'].unsqueeze(0) for example in examples]),
            'labels-mtr-mask': torch.cat([example['labels-mtr-mask'].unsqueeze(0) for example in examples]),
        }

        smi_mtr_tokenized = self.smi_tokenizer.batch_encode_plus(
            [
                example['smiles']
                for example in examples
            ],
            return_special_tokens_mask=True,
        )

        batch_smi = smi_mtr_tokenized

        # If special token mask has been preprocessed, pop it from the dict.
        # special_tokens_mask_txt = batch_txt.pop("special_tokens_mask", None)
        # assert special_tokens_mask_txt is not None, "Special tokens mask must be provided."

        # batch_txt['input-mlm'], batch_txt['labels-mlm'] = self.torch_mask_tokens(
        #     inputs=batch_txt['input-cl'],
        #     special_tokens_mask=special_tokens_mask_txt,
        #     mask_id=self.txt_tokenizer.convert_tokens_to_ids(self.txt_tokenizer.mask_token),
        #     vocab_size=len(self.txt_tokenizer), 
        #     mlm_probability=self.mlm_probability,
        # )

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask_smi = batch_smi.pop("special_tokens_mask", None)
        assert special_tokens_mask_smi is not None, "Special tokens mask must be provided."

        # batch_smi['input-mlm'], batch_smi['labels-mlm'] = self.torch_mask_tokens(
        #     inputs=batch_smi['input'],
        #     special_tokens_mask=special_tokens_mask_smi,
        #     mask_id=self.smi_tokenizer.mask_id,
        #     vocab_size=self.smi_tokenizer.vocab_size, 
        #     mlm_probability=self.mlm_probability,
        # )

        return {
            'txt': batch_txt,
            'smi': batch_smi
        }
    
    def torch_mask_tokens(self, 
                          inputs: Any, 
                          special_tokens_mask: Optional[Any] = None, 
                          mask_id: int = 1, 
                          vocab_size: int = 30522,
                          mlm_probability: float = 0.15) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)

        special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = mask_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels