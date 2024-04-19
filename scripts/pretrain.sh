#!/bin/zsh

source ~/.zshrc

conda activate st

python ../pretrain/main.py \
--seed                  42 \
--batch_size            64 \
--num_workers           6 \
--lr                    5.5e-05 \
--warmup_proportion     0.1 \
--max_epochs            50 \
--devices               2 \
--precision             "bf16-mixed" \
--overfit_batches       0.0 \
--use_tune              0 \
--batch_size_find       1 \
--lr_find               0 \
--n_trials              10 \
--limit_train_batches   0.05 \
--limit_val_batches     0.1 \
--logger_offline        0 \
--data_name             "mt-mtr.pt"
