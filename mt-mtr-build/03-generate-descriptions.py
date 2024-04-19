import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
data_dir = base_dir / 'data'


import os
import time
import json
import random

import openai
import pandas as pd

from pandarallel import pandarallel

import argparse

# openai api-key is contained in the key.json which has been removed
with open(data_dir / 'key.json', 'r') as f:
    key = json.load(f)

os.environ["http_proxy"]    = f"http://127.0.0.1:7890"
os.environ["https_proxy"]   = f"http://127.0.0.1:7890"

openai.api_key = key['api-key']
openai.organization = key['organization']



def generate_response_base(input):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': f'{input}'}],
        # max_tokens=15,
    )
    return response['choices'][0]['message']['content']


def get_task_description(sample_dict):
    instruction = 'As a seasoned expert in the field of chemistry, your task is to analysis a chemical task. And you found following properties of chemical compound can help solve this task. Please summarize your analyse. The length should be less than 300 tokens.'
    prompt = f'{instruction}\n\nDescriptors:\n{json.dumps(list(sample_dict.keys()), indent=4)}\n\nTask analysis results:'

    return generate_response_base(prompt)


df = pd.read_json(data_dir / 'temporary/descriptors.jsonl', lines=True, orient='records')
ratio_path = data_dir / 'temporary/ratio.json'
with open(ratio_path, 'r') as f:
    ratio_dict = json.load(f)


def generate(series: pd.Series):
        save_dict = {
            'smiles': series['smiles'],
            'descriptors': '',
            'task_description': '',
        }

        desc_dict = series.to_dict()
        desc_dict.pop('smiles')

        # if fr_ prefix in sample_key and the value is zero, the ratio of reject is the value of ratio_dict
        # while until len of sample_dict is sample_num
        sample_num = random.randint(5, 10)
        sample_keys = []
        while True:
            sample_keys = random.sample(list(desc_dict.keys()), sample_num)
            sample_dict = {k: desc_dict[k] for k in sample_keys}
            flag = False

            for k, v in sample_dict.items():
                if v == 0 and k.startswith('fr_'):
                    if random.random() < ratio_dict[k]:
                        flag = True
                        break
            if flag:
                continue
            break


        # print(f'task description')
        while True:
            try:
                task_description =  get_task_description(sample_dict)
                break
            except Exception as e:
                print(e)
                print('try again')
                time.sleep(2)
        
        save_dict['descriptors'] = json.dumps(sample_dict)
        save_dict['task_description'] = task_description

        return pd.Series(save_dict)


def merge():
    df_list = []
    for i in range(1000, 57000, 1000):
        df = pd.read_json(data_dir / f'temporary/descriptions-{i:05d}.jsonl', lines=True)
        df_list.append(df)
    
    df = pd.concat(df_list, ignore_index=True)
    df.to_json(data_dir / 'mt-mtr-origin.jsonl', orient='records', lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num', type=int, default=0)

    args = parser.parse_args()
    num = args.num

    df_slice = df.iloc[0+num:1000+num]
    print(df_slice.shape)

    pandarallel.initialize(progress_bar=True, nb_workers=10)

    # df.apply(func)
    save = df_slice.parallel_apply(generate, axis=1)
    save.to_json(data_dir / f'temporary/descriptions-{1000+num:>05}.jsonl', orient='records', lines=True)

    merge()
