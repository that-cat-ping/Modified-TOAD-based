# -*- coding: UTF-8 -*-
import os
import random
import shutil
from glob import glob 
import pandas as pd
import numpy as np
import argparse
import yaml
from pathlib import Path  
import time
from sklearn.model_selection import StratifiedKFold 


def useCrossValidation(X, y, split_path, K=5):
    print('K_fold:',K)
    skf = StratifiedKFold(n_splits=K, shuffle=True)
    for fold, (train, test) in enumerate(skf.split(X, y)):
        train_data = {}
        test_data = {}
        train_set, train_label = pd.Series(X).iloc[train].tolist(), pd.Series(y).iloc[train].tolist()
        test_set, test_label = pd.Series(X).iloc[test].tolist(), pd.Series(y).iloc[test].tolist()
        train_data['id'] = train_set
        train_data['label'] = train_label
        test_data['id'] = test_set
        test_data['label'] = test_label
        train = pd.DataFrame(train_data)
        test = pd.DataFrame(test_data)
        train.to_csv(split_path+f'train_{fold}.csv', index=False)
        test.to_csv(split_path+f'test_{fold}.csv', index=False)


def split_transform(Original_address, transform_address, K=5):
    for i in range(0,K):
        train_file = pd.read_csv(Original_address+f'train_{i}.csv')
        test_file = pd.read_csv(Original_address+f'test_{i}.csv')
        train_file = train_file.dropna(axis=0, how='all')
        test_file = test_file.dropna(axis=0, how='all')
        train_id = train_file['id'].tolist()
        train_label = train_file['label'].tolist()
        test_id = test_file['id'].tolist()
        test_label = test_file['label'].tolist()

        train_id_df = pd.DataFrame(train_id)
        test_id_df = pd.DataFrame(test_id)
        splits_k = pd.concat([train_id_df,test_id_df], ignore_index=True, axis=1)
        splits_k.columns = ['train','test']

        dict_splits_k_bool = {}
        id = train_id + test_id
        for m in range(len(id)):
            if id[m] in train_id:
                dict_splits_k_bool[id[m]] = {"train":"TRUE","test":"FALSE"}
            elif id[m] in test_id:
                dict_splits_k_bool[id[m]] = {"train": "FALSE","test":"TRUE"}
        splits_k_bool = pd.DataFrame(dict_splits_k_bool).T

        dict_splits_k_descriptor={}
        l_1=0
        l_0=0
        for n in range(len(train_label)):
            if train_label[n] == 1:
                l_1 += 1
            else:
                l_0 += 1
        t_0 = 0
        t_1 = 0
        for v in range(len(test_label)):
            if test_label[v] == 1:
                t_1 += 1
            else:
                t_0 += 1
        dict_splits_k_descriptor["0"]={"train":l_0,"test":t_0}
        dict_splits_k_descriptor["1"] = {"train":l_1, "test":t_1}
        print(dict_splits_k_descriptor)
        splits_k_descriptor = pd.DataFrame(dict_splits_k_descriptor).T

        splits_k.to_csv(transform_address+f"splits_{i}.csv")
        splits_k_bool.to_csv(transform_address+f"splits_{i}_bool.csv")
        splits_k_descriptor.to_csv(transform_address+f"splits_{i}_descriptor.csv")


def main(label,K,split_path,transform_path,shuffle=True):
    assert os.path.exists(label), "Error: 标签文件不存在"
    assert Path(label).suffix == '.csv', "Error: 标签文件需要是csv文件"
    try:
        df = pd.read_csv(label, usecols=["slide_id", "label"])
    except :
        print("Error: 未在文件中发现ID或标签列信息")
    ID = df['slide_id'].tolist()
    Label = df['label'].tolist()
    useCrossValidation(ID, Label, split_path,K)
    split_transform(split_path,transform_path,K)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script', epilog="authorized by lvyp ")
    parser.add_argument('--label_dir_path', type=str, default="/home/lvyp/fuzhong_project/toad_test/csv_file//Data_after_matching.csv")
    parser.add_argument('--K_fold_split_path', type=str, default="/home/lvyp/fuzhong_project/toad_test/5_fold_split/")
    parser.add_argument('--transform_path', type=str, default="/home/lvyp/fuzhong_project/toad_test/split/")
    parser.add_argument('--K_fold', type=int, default=5)
    args = parser.parse_args()
    main(args.label_dir_path,args.K_fold, args.K_fold_split_path,args.transform_path)
        