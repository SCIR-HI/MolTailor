#!/bin/zsh

source ~/.zshrc

conda activate moltailor

if [ ! -d "../pretrain/workspace" ]; then
  mkdir "../pretrain/workspace"
fi

python ../pretrain/main.py \
--seed                  42 \
--batch_size            64 \
--num_workers           6 \
--lr                    5.5e-05 \
--warmup_proportion     0.1 \
--max_epochs            50 \
--devices               2 \
--precision             "bf16-mixed" \
--use_tune              0 \
--logger_offline        0 \
--model_name_txt        "PubMedBERT" \
--model_name_smi        "CHEM-BERT" \
--data_name             "mt-mtr.pt"
