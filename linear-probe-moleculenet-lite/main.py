'''
Author: simwit 517992857@qq.com
Date: 2023-07-24 21:58:05
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-07 18:54:20
FilePath: /hqguo/workspace/01-st/finetune-moleculenet/main.py
Description: 

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir.parent))


import argparse
import numpy as np

from timeit import default_timer as timer

from multi_seeds import MultiSeeedsFinetune


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # args for dataset
    parser.add_argument('--feature_name', type=str, default='ChemBERTa-77M-MTR')
    parser.add_argument('--task_name', type=str, default='bbbp')

    # args for data_module
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--inference_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)

    # args for trainer and tuner
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--root_dir', type=str, default='workspace')
    parser.add_argument('--devices', type=str, default='auto')

    # args for trainer
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--logger_name', type=str, default='')
    
    # args for tuner
    parser.add_argument('--use_tune', type=int, default=1)
    parser.add_argument('--lr_max', type=float, default=1e-2)
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--n_trials', type=int, default=10)

    # args for multi_seeds
    parser.add_argument('--seeds', type=str, default='1236, 1237, 1238')

    # args for reassification
    parser.add_argument('--use_reassification', type=int, default=0)


    args = parser.parse_args()
    args.root_dir = str(base_dir / args.root_dir)
    args.use_tune = bool(args.use_tune)
    args.use_reassification = bool(args.use_reassification)


    start_time = timer()

    finetune = MultiSeeedsFinetune(
        feature_name=args.feature_name,
        task_name=args.task_name,
        train_batch_size=args.train_batch_size,
        inference_batch_size=args.inference_batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        root_dir=args.root_dir,
        devices=args.devices,
        lr=args.lr,
        logger_name=args.logger_name,
        use_tune=args.use_tune,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        n_trials=args.n_trials,
        seeds=[int(ele) for ele in args.seeds.split(',')],
        use_reassification=args.use_reassification,
    )

    results = finetune.run()

    end_time = timer()

    

    
    print(f'---   id    --- \n{", ".join(results["run_id"])}')
    print(f'[{args.feature_name}]-{args.task_name}')
    if finetune.task_type == 'classification':
        print(f'--- roc_auc --- \n{np.mean(results["roc_auc"])*100:.2f}\n{np.std(results["roc_auc"])*100:.2f}')
        print(f'---   dap   --- \n{np.mean(results["dap"])*100:.2f}\n{np.std(results["dap"])*100:.2f}')
    elif finetune.task_type == 'regression':
        print(f'---  rmse   --- \n{np.mean(results["rmse"]):.6f}\n{np.std(results["rmse"]):.6f}')
    print(f'Train_time (sec): \n{np.mean(results["train_time"]):.2f}\n{np.std(results["train_time"]):.2f}')
    print(f'Elapsed time (min): \n{(end_time - start_time)/60:.2f}')
    
