import sys
import runpy
import os

base_path = os.path.dirname(__file__)
os.chdir(base_path)

args = """
python main.py \
--seed                  42 \
--batch_size            2 \
--num_workers           2 \
--lr                    5.5e-05 \
--warmup_proportion     0.1 \
--max_epochs            2 \
--devices               1 \
--overfit_batches       0.01 \
--use_tune              0 \
--batch_size_find       1 \
--lr_find               0 \
--n_trials              10 \
--limit_train_batches   0.05 \
--limit_val_batches     0.1 \
--logger_offline        1 \
--model_name_txt        BioLinkBERT \
--data_name             mt-mtr.pt
"""


args = args.split()
if args[0] == 'python':
    """pop up the first in the args"""
    args.pop(0)

if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

sys.argv.extend(args[1:])

fun(args[0], run_name='__main__')