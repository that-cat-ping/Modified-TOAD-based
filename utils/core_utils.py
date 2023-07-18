import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
#from models.model_clam_tt import TOAD_mtl, CLAM_SB
from models.model_clam_mcb import TOAD_mtl, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve,auc
from sklearn.metrics import auc as calc_auc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
       
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count  
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=100, stop_epoch=300, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

class FocalLoss(nn.Module):
    def __init__(self, num_labels=2, activation_type='softmax', gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type
        self.num_labels = num_labels

    def forward(self, input, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(input, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = torch.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (
                    1 - logits + self.epsilon).log()
        return loss.mean()


def auc_curve(y, prob, cur, color='navy', title='ROC curve'):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    frame_1 = pd.DataFrame({"tpr": tpr, "fpr": fpr})
    frame_1.to_csv(os.path.join("/mnt/breast196/results/result_去掉M1和MX_s5/tpr_fpr_{}.csv".format(cur)), index=False)

    precision, recall, _ = precision_recall_curve(y, prob)
    frame_2 = pd.DataFrame({"precision": precision, "recall": recall})
    #frame_2.to_csv(os.path.join("/mnt/breast196/results/result_去掉M1和MX_s5/pre_recall_{}.csv".format(cur)), index=False)
    
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color=color, lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join("/mnt/breast196/results/result_去掉M1和MX_s5/ROC_{}.pdf".format(cur)))
    #plt.show()
    return  roc_auc

def confusion_matrix(labels, probs):
    '''print("all_labels", all_labels)
        print("all_probs[:, 1]", all_probs[:, 1])
        auc = auc_curve(all_labels, all_probs[:, 1],cur)'''

    y_test_pred_binary = []
    for item in probs:
        if item <= 0.5:
            a = 0
        else:
            a = 1
        y_test_pred_binary.append(a)
    accuracy = accuracy_score(labels, y_test_pred_binary)
    precision = precision_score(labels, y_test_pred_binary)
    recall = recall_score(labels, y_test_pred_binary)
    f1 = f1_score(labels, y_test_pred_binary)
    result_list = [accuracy,precision,recall,f1]
    return result_list

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    if args.log_data:
        from tensorboardX import SummaryWriter  # 可视化
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None
    print('\nInit train/test splits...', end=' ')
    train_split,val_split, test_split = datasets
    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    #损失函数
    loss_fn = nn.CrossEntropyLoss()
    #weight = [0.1,0.9]
    #class_weights = torch.FloatTensor(weight).to(device)
    #loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    #loss_fn = FocalLoss()
    #loss_fn = nn.BCELoss()

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        instance_loss_fn = nn.CrossEntropyLoss()
        #instance_loss_fn = FocalLoss()
        if args.model_type == 'clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    else:
        model = TOAD_mtl(**model_dict)

    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    #梯度下降法
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 100, stop_epoch=300, verbose = True)
    else:
        early_stopping = None
    print('Done!')
    
    all_train_loss =[]
    all_valid_loss =[]
    for epoch in range(args.max_epochs):
        print("epoch:  ",epoch)

        if args.model_type in ['clam_sb', 'clam_mb']:
            train_loss=train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            valid_loss,stop = validate_clam(cur, epoch, model, val_loader, args.n_classes,
                                 early_stopping, writer, loss_fn, args.results_dir)
            all_train_loss.append(train_loss)
            all_valid_loss.append(valid_loss)

        else:
            train_loss = train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            valid_loss, stop = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn,
                                        args.results_dir)
            all_train_loss.append(train_loss)
            all_valid_loss.append(valid_loss)

        if stop: 
            break
    final_loss = pd.DataFrame({'train_loss': all_train_loss, 'valid_loss': all_valid_loss})
    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, valid_error, valid_auc, _,_= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(valid_error, valid_auc))
    results_dict, test_error, test_auc, acc_logger,result_df = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
    if writer:
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
    writer.close()
    return results_dict, test_auc, 1-test_error,result_df,final_loss

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None,):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    print('\n')
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    for batch_idx, (data, label,other_feature) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        other_feature = other_feature.float().to(device)
        optimizer.zero_grad()
        results_dict = model(data,other_feature)
        logits, Y_prob, Y_hat  = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
        prob[batch_idx] = Y_prob.cpu().detach().numpy() 
        labels[batch_idx] = label.item()
        print("label and Y_prob：",label, Y_prob, "\n")
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()     
        train_loss += loss_value
        if (batch_idx + 1) % 5 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
        error = calculate_error(Y_hat, label)
        train_error += error        
        # backward pass
        loss.backward()
        optimizer.step()        
    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    auc = roc_auc_score(labels, prob[:, 1])
    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f} ,auc: {:.4f}'.format(epoch, train_loss, train_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
    return train_loss

def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    for batch_idx, (data, label,other_feature) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        other_feature = other_feature.float().to(device)
        optimizer.zero_grad()
        results_dict = model(data,other_feature,label=label, instance_eval=True)
        logits, Y_prob, Y_hat  = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
        prob[batch_idx] = Y_prob.cpu().detach().numpy()
        labels[batch_idx] = label.item()
        print("label and Y_prob：",label, Y_prob, "\n")

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        print(results_dict)
        instance_loss = results_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss

        inst_preds = results_dict['inst_preds']
        inst_labels = results_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 5 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, label: {}, bag_size: {}'.
                  format(batch_idx, loss_value, instance_loss_value, total_loss.item(), label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    auc = roc_auc_score(labels, prob[:, 1])

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f},train_clustering_loss: {:.4f}, train_error: {:.4f} ,auc: {:.4f}'.
          format(epoch, train_loss, train_inst_loss, train_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
    return train_loss

def validate(cur, epoch, model, loader,n_classes,  early_stopping = None, writer = None,loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    valid_loss = 0.
    valid_error = 0.
    print('\n')
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, label, other_feature) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            other_feature = other_feature.float().to(device)
            results_dict = model(data, other_feature)
            logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            loss_value = loss.item()
            prob[batch_idx] = Y_prob.cpu().detach().numpy()
            labels[batch_idx] = label.item()
            valid_loss += loss_value
            error = calculate_error(Y_hat, label)
            valid_error += error
        valid_loss /= len(loader)
        valid_error /= len(loader)
        auc = roc_auc_score(labels, prob[:, 1])
        if writer:
            writer.add_scalar('valid/loss', valid_loss, epoch)
            writer.add_scalar('valid/error', valid_error, epoch)
        print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(valid_loss,valid_error, auc))
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            if writer:
                writer.add_scalar('valid/class_{}_acc'.format(i), acc, epoch)
        if early_stopping:
            assert results_dir
            early_stopping(epoch, valid_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
            if early_stopping.early_stop:
                print("Early stopping")
                return valid_loss,True
        return valid_loss,False

def validate_clam(cur, epoch, model, loader,n_classes,  early_stopping = None, writer = None,loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    valid_loss = 0.
    valid_error = 0.
    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count = 0

    total_loss = 0

    print('\n')
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label, other_feature) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            other_feature = other_feature.float().to(device)
            results_dict = model(data,  other_feature,label=label,instance_eval=True)
            logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)
            loss_value = loss.item()
            prob[batch_idx] = Y_prob.cpu().detach().numpy()
            labels[batch_idx] = label.item()
            valid_loss += loss_value

            instance_loss = results_dict['instance_loss']

            inst_count += 1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = results_dict['inst_preds']
            inst_labels = results_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            error = calculate_error(Y_hat, label)
            valid_error += error

        valid_loss /= len(loader)
        valid_error /= len(loader)

        total_loss /= len(loader)

        auc = roc_auc_score(labels, prob[:, 1])
        if writer:
            writer.add_scalar('valid/loss', valid_loss, epoch)
            writer.add_scalar('valid/error', valid_error, epoch)

        print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(valid_loss,valid_error, auc))
        if inst_count > 0:
            val_inst_loss /= inst_count
            for i in range(2):
                acc, correct, count = inst_logger.get_summary(i)
                print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            if writer:
                writer.add_scalar('valid/class_{}_acc'.format(i), acc, epoch)

        if early_stopping:
            assert results_dir
            early_stopping(epoch, valid_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
            if early_stopping.early_stop:
                print("Early stopping")
                return valid_loss,True
        return valid_loss,False


def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, other_feature) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        other_feature = other_feature.float().to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            results_dict = model(data, other_feature)
        logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
        del results_dict
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error
    test_error /= len(loader)
    print("all_labels", all_labels)
    print("all_probs[:, 1]", all_probs[:, 1])
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    result_list = confusion_matrix(all_labels, all_probs[:, 1])
    result_list.append(auc)

    return patient_results, test_error, auc, acc_logger,result_list
