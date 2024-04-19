import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir.parent))

import argparse

from timeit import default_timer as timer

import torch
import numpy as np

from train import MultiTaskPreTrainingTrainer
from tune import MultiTaskPreTrainingTuner


if __name__ == '__main__':
    # torch.set_float32_matmul_precision('medium') # set for NVIDIA A100-SXM4-80GB
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)


    # args for data_module
    parser.add_argument('--model_name_txt', type=str, default='PubMedBERT')
    parser.add_argument('--model_name_smi', type=str, default='CHEM-BERT')
    parser.add_argument('--data_name', type=str, default='multi-task-pretrain.pt')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)

    # args for model
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)

    # args for trainer and tuner
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32-true')
    parser.add_argument('--root_dir', type=str, default='workspace')
    parser.add_argument('--overfit_batches', type=float, default=0.0)
    parser.add_argument('--logger_offline', type=int, default=0)
    
    parser.add_argument('--if_train', type=int, default=1)

    # args for tuner
    parser.add_argument('--use_tune', type=int, default=1)
    parser.add_argument('--batch_size_find', type=int, default=1)
    parser.add_argument('--lr_find', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--limit_train_batches', type=float, default=0.01)
    parser.add_argument('--limit_val_batches', type=float, default=0.1)


    args = parser.parse_args()
    args.root_dir = str(base_dir / args.root_dir)
    args.if_train = bool(args.if_train)
    args.use_tune = bool(args.use_tune)
    args.batch_size_find = bool(args.batch_size_find)
    args.lr_find = bool(args.lr_find)
    args.logger_offline = bool(args.logger_offline)

    start_time = timer()

    if args.use_tune:
        tuner = MultiTaskPreTrainingTuner(
            seed=args.seed,
            model_name_txt=args.model_name_txt,
            model_name_smi=args.model_name_smi,
            data_name=args.data_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            devices=args.devices,
            precision=args.precision,
            warmup_proportion=args.warmup_proportion,
            max_epochs=args.max_epochs,
            root_dir=args.root_dir,
            batch_size_find=args.batch_size_find,
            lr_find=args.lr_find,
            n_trials=args.n_trials,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )
        outputs = tuner.run()
        if args.batch_size_find:
            args.batch_size = outputs['optimal_batch_size']
        if args.lr_find:
            args.lr = outputs['optimal_lr']
    
    trainer = MultiTaskPreTrainingTrainer(
        seed=args.seed,
        model_name_txt=args.model_name_txt,
        model_name_smi=args.model_name_smi,
        data_name=args.data_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        warmup_proportion=args.warmup_proportion,
        root_dir=args.root_dir,
        overfit_batches=args.overfit_batches,
        logger_offline=args.logger_offline,
    )

    if args.if_train:
        trainer.run()
    
    end_time = timer()

    print(f'Elapsed time: {(end_time - start_time) / 60:.2f} min')