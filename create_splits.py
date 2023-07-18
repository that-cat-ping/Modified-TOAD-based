import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default=1.0,help='fraction of labels (default: 1)')
parser.add_argument('--csv_path', type=str, default=None,help = 'label csv directory')
parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use,'+'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--seed', type=int, default=1,help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=5,help='number of splits (default: 10)')
args = parser.parse_args()

args.n_classes = 2
dataset = Generic_WSI_Classification_Dataset(csv_path= args.csv_path,
                                             shuffle=False,
                                             seed=args.seed,
                                             print_info=True,
                                             label_dicts={'0':0, '1':1},
                                             label_cols = ['label'],
                                             patient_strat=False)

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.floor(num_slides_cls * 0.2).astype(int)
test_num = np.floor(num_slides_cls * 0.2).astype(int)
print(val_num)
print(test_num)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    for lf in label_fracs:
        split_dir = args.split_dir + 'train_valid_test_new' + '_{}'.format(args.k)
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            if dataset.split_gen is None:
                ids = []
                for split in ['train', 'val', 'test']:
                    ids.append(
                        dataset.get_split_from_df(pd.read_csv(os.path.join(split_dir, 'splits_{}.csv'.format(i))),
                                                  split_key=split, return_ids_only=True))

                dataset.train_ids = ids[0]
                dataset.val_ids = ids[1]
                dataset.test_ids = ids[2]
            else:
                dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



