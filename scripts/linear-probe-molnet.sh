#!/bin/zsh

source ~/.zshrc

conda activate st


# 1 feature extract
model_name=$1
prompt_file=$2

# if prompt_file is empty，use default prompt_file
if [ -z $prompt_file ]
then
    prompt_file='prompt4molnet.json'
fi
# 输出model_name和prompt_file
echo "model_name: $model_name"
echo "prompt_file: $prompt_file"


for task_name in 'bbbp' 'clintox' 'hiv' 'tox21' 'esol' 'freesolv' 'lipophilicity' 'qm8'
do
    python ../linear-probe-moleculenet/data/utils/feature-extract.py --model_name $model_name --prompt_file $prompt_file --task_name $task_name
    python ../linear-probe-moleculenet/main.py --task_name $task_name --feature_name $model_name
done