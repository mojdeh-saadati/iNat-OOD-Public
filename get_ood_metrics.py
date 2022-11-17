import numpy as np;
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import sklearn.metrics as sk
from collections import OrderedDict
import nbimporter
import argparse 

device = 'cuda'



def return_OOD_metrics_info(path, risky = None , safe = None):
    
    if((len(risky) == 0) and (len(safe) == 0)):
        start = time.time();
        safe, risky = torch.load(path);  
        print("safe.type", type(safe))
        if not type(safe) is np.ndarray:
            safe, risky = safe.cpu().numpy(), risky.cpu().numpy()  

        if len(safe.shape) == 1:
            safe = safe.reshape((safe.shape[0],1))
        if len(risky.shape) == 1:
            risky = risky.reshape((risky.shape[0],1))
        print("safe contrct:",safe[0:2,:])
        print("risky contentL", risky[0:2,:])    
        if(safe.shape[0] > risky.shape[0]):
            safe = safe[0:risky.shape[0],]
        else:
            risky = risky[0:safe.shape[0],]

    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    examples = np.squeeze(np.vstack((safe, risky)))
    print("label.shape ===", labels.shape)
    print("example.shape ===", examples.shape)
    labels[0:safe.shape[0]] += 1
    AUPR = round(100*sk.average_precision_score(labels, examples), 2)
    AUROC =  round(100*sk.roc_auc_score(labels, examples), 2)
    _,_, fpr95 = get_curve(safe, risky)
    print(AUROC, AUPR, fpr95)
    return [AUROC, AUPR, fpr95]


def get_curve(known, novel, method=None):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")

    return parser

"""
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    return_OOD_metrics_info(args.path) 
"""