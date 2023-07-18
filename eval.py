from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from math import floor
import h5py
import matplotlib.pyplot as plt

from utils.utils import *
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,help='data directory')
parser.add_argument('--csv_path', type=str, default=None,help = 'label csv directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,help='experiment code to save eval results')
parser.add_argument('--splits_dir', type=str, default=None, help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False, help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.save_dir = os.path.join('./eval_results', str(args.save_exp_code))
args.models_dir = args.results_dir
os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir
assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'csv_path':args.csv_path,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'micro_avg': args.micro_average}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
args.n_classes=2
dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                              data_dir= args.data_root_dir,
                              shuffle = False, 
                              print_info = True,
                              label_dicts = {'0':0, '1':1},
                              label_cols = ['label'],
                              patient_strat=False,)
if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end
if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}


if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        '''if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]'''
            
        split_dataset = dataset
        model, results_dict = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_auc.append(results_dict['auc_score'])
        all_acc.append(1-results_dict['test_error'])
        df = results_dict['df']
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    df_dict = {'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc}
    final_df = pd.DataFrame(df_dict)
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
