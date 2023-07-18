import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from models.compact_bilinear_pooling import CountSketch, CompactBilinearPooling


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D),nn.Tanh()]
        self.attention_b = [nn.Linear(L, D),nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
args: 
    gata:whether to use gated attention network
    size_arg:config for network size
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "big", dropout = False, k_sample=8, n_classes=2, use_other_feature=True,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, method="mcbp"):
        super(CLAM_SB, self).__init__()

        self.size_dict = {"small": [256, 256, 128], "big": [1024, 512, 256]}
        self.method = method
        size = self.size_dict[size_arg]
        self.use_other_feature = use_other_feature
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        if self.use_other_feature:
            fc_ = [nn.Linear(7, 7)]
            #if dropout:
            #    fc_.append(nn.Dropout(0.25))
            self.other_feature = nn.Sequential(*fc_)
            if method == "mcbp":
                self.mcb = CompactBilinearPooling(size[1], 7, size[1])
                #self.relu = nn.ReLU()
                #if dropout:
                    #self.dropout = nn.Dropout(0.1)
                self.classifiers1 = nn.Linear(size[1], 256)
                self.classifiers2 = nn.Linear(256, 128)
                self.classifiers3 = nn.Linear(128,64)
                self.classifiers = nn.Linear(64, n_classes)
            elif method == "concat":
                self.concat = nn.Linear(size[1] + 803, size[1])
                #self.relu = nn.ReLU()
                #if dropout:
                    #self.dropout = nn.Dropout(0.25)
                self.classifiers1 = nn.Linear(size[1], 256)
                self.classifiers2 = nn.Linear(256, 128)
                self.classifiers3 = nn.Linear(128,64)
                self.classifiers = nn.Linear(64, n_classes)
            else:
                raise NotImplementedError
        else:
            self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        if self.use_other_feature:
            self.other_feature = self.other_feature.to(device)
            if self.method == "mcbp":
                self.mcb = self.mcb.to(device)
            elif self.method == "concat":
                self.concat = self.concat.to(device)
            # self.relu = self.relu.to(device)
            # self.dropout = self.dropout.to(device)
            self.classifiers1 = self.classifiers1.to(device)
            self.classifiers2 = self.classifiers2.to(device)
            self.classifiers3 = self.classifiers3.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, omis_feature, label=None, instance_eval=False, return_features=False, use_other_feature=True, attention_only=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
        M = torch.mm(A, h)
        if self.use_other_feature:
            feature = self.other_feature(omis_feature.to(device))
            # print(M)
            if self.method == "mcbp":
                M = self.mcb(M, feature.repeat(M.size(0), 1))
                # M = self.relu(M)
                # M = self.dropout(M)
                logits1 = self.classifiers1(M[0].unsqueeze(0))
                logits2 = self.classifiers2(logits1)
                logits3 = self.classifiers3(logits2)
                logits = self.classifiers(logits3)
            elif self.method == "concat":
                M = torch.cat([M, feature.repeat(M.size(0), 1)], dim=1)
                M = self.concat(M)
                #M = self.relu(M)
                # M = self.dropout(M)
                #logits_ = self.classifiers1(M[0].unsqueeze(0))
                logits1 = self.classifiers1(M[0].unsqueeze(0))
                logits2 = self.classifiers2(logits1)
                logits3 = self.classifiers3(logits2)
                logits = self.classifiers(logits3)
                #logits = self.classifiers(logits_)
            else:
                raise NotImplementedError
        else:
            logits = self.classifiers(M[0].unsqueeze(0))
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        results_dict.update({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'A': A_raw})
        return results_dict


"""
TOAD multi-task + concat mil network w/ attention-based pooling
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class TOAD_mtl(nn.Module):
    def __init__(self, gate = True, size_arg = "big", dropout = False, n_classes=2, other_feature=False):
        super(TOAD_mtl, self).__init__()
        self.size_dict = {"small": [256, 256, 128], "big": [1024, 512, 256]}
        size = self.size_dict[size_arg]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        if other_feature:
            fc_ = [nn.Linear(0, 128), nn.ReLU()]
            if dropout:
                fc_.append(nn.Dropout(0.25))
            self.other_feature = nn.Sequential(*fc_)
            if method == "mcbp":
                self.mcb = CompactBilinearPooling(size[1], 128, size[1])
                self.relu = nn.ReLU()
                if dropout:
                    self.dropout = nn.Dropout(0.1)
                self.classifiers = nn.Linear(size[1], n_classes)
            elif method == "concat":
                self.concat = nn.Linear(size[1] + 128, size[1])
                self.relu = nn.ReLU()
                if dropout:
                    self.dropout = nn.Dropout(0.25)
                self.classifiers = nn.Linear(size[1], n_classes)
            else:
                raise NotImplementedError
        else:
            self.classifiers = nn.Linear(size[1], n_classes)
        initialize_weights(self)


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        if self.method == "mcbp":
            self.mcb = self.mcb.to(device)
        elif self.method == "concat":
            self.concat = self.concat.to(device)
        self.relu = self.relu.to(device)
        self.dropout = self.dropout.to(device)
        self.classifiers = self.classifiers.to(device)


    def forward(self, h, omis_feature, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK      
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)
        feature = self.other_feature(omis_feature.to('cuda'))
        if self.method == "mcbp":
            M = self.mcb(M, feature.repeat(M.size(0), 1))
            M = self.relu(M)
            M = self.dropout(M)
        elif self.method == "concat":
            M = torch.cat([M, feature.repeat(M.size(0), 1)], dim=1)
            M = self.concat(M)
            M = self.relu(M)
            M = self.dropout(M)
        else:
            raise NotImplementedError
        logits  = self.classifiers(M[0].unsqueeze(0))
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {}
        if return_features:
            results_dict.update({'features': M})           
        results_dict.update({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'A': A_raw})
        return results_dict

