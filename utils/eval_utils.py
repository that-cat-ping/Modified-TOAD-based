import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from models.model_clam_mcb import TOAD_mtl, CLAM_SB
from utils.utils import *
from utils.core_utils import Accuracy_Logger


def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type in ['clam_sb','clam_mb']:
        if args.model_type == 'clam_sb':
            model = CLAM_SB(**model_dict)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict)
        else:
            raise NotImplementedError
    else:
        model = TOAD_mtl(**model_dict)
    print_network(model)
    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)
    model.relocate()
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    results_dict = summary(model, loader, args)
    print('test_error: ', results_dict['test_error'])
    print('auc: ', results_dict['auc_score'])
    return model, results_dict

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label, other_feature) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        other_feature = other_feature.float().to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            results_dict = model(data, other_feature)
        logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error
    del data
    test_error /= len(loader)
    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1
    else:
        auc_score = roc_auc_score(all_labels, all_probs[:, 1])
    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    inference_results = {'patient_results': patient_results, 'test_error': test_error, 'auc_score': auc_score,
                         'loggers': acc_logger, 'df': df}
    return inference_results
