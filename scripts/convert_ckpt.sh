#!/bin/zsh

source ~/.zshrc

conda activate st

ckpt_id=$1

model_name='last.ckpt'
save_name='last.ckpt'

source_path='../pretrain/workspace/checkpoints/'$ckpt_id'/'
target_path='../models/DEN/'$ckpt_id'/'

mkdir $target_path
cp $source_path$model_name $target_path$save_name

python ../pretrain/models/multitask.py --ckpt_id $ckpt_id