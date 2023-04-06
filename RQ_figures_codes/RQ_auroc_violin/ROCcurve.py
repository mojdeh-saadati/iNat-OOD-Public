import numpy as np;
import matplotlib.pyplot as plt
import time
import matplotlib
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
from sklearn.metrics import roc_curve, auc
import pandas as pd;
import seaborn as sns;

device = 'gpu'
#################################################################
# return_OOD_metrics_info in the first step takes the result vectors 
# if they are empty then it can read if from the input path. 
##################################################################
def return_OOD_metrics_info(path, risky = None , safe = None):
    
    if((len(risky) == 0) and (len(safe) == 0)):
        start = time.time();
        safe, risky = torch.load(path);  
        if not type(safe) is np.ndarray:
            safe, risky = safe.cpu().numpy(), risky.cpu().numpy()  

        if len(safe.shape) == 1:
            safe = safe.reshape((safe.shape[0],1))
        if len(risky.shape) == 1:
            risky = risky.reshape((risky.shape[0],1))
        if(safe.shape[0] > risky.shape[0]):
            safe = safe[0:risky.shape[0],]
        else:
            risky = risky[0:safe.shape[0],]

    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    examples = np.squeeze(np.vstack((safe, risky)))
    labels[0:safe.shape[0]] += 1
    AUPR = round(100*sk.average_precision_score(labels, examples), 2)
    AUROC =  round(100*sk.roc_auc_score(labels, examples), 2)
    fpr, tpr, _ = sk.roc_curve(labels, examples)

    matplotlib.rc('xtick', labelsize=15) 
    matplotlib.rc('ytick', labelsize=15) 
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/RQ_figures_notebooks/RQ4/roc_curve.pdf", bbox_inches = 'tight')
    plt.close()

    red_circle = dict(markerfacecolor='red', marker='o')
    plt.yscale('log')
    c = 'skyblue'
    d = 'darkblue'
    plt.boxplot(risky.flatten(), positions = [1],
                 patch_artist=True, 
                boxprops=dict(facecolor=c, color=c),
                medianprops=dict(color=d), labels = ["OOD"]                
                )    

    plt.boxplot(safe.flatten(), positions = [2],
                 patch_artist=True, 
                boxprops=dict(facecolor=d, color=d),
                medianprops=dict(color=c),
                labels = ["ID"]
                          
                )       
    quantile1 = list(np.quantile(safe.flatten(), np.array([0.00, 0.25, 0.50, 0.75, 1.00])))
    quantile2 = list(np.quantile(risky.flatten(), np.array([0.00, 0.25, 0.50, 0.75, 1.00])))
    matplotlib.rc('xtick', labelsize=15) 
    matplotlib.rc('ytick', labelsize=15)                                                                          
    plt.xlabel("Dataset Category", fontsize=15)
    plt.ylabel("Energy", fontsize=15)
    plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/RQ_figures_notebooks/RQ4/ID_OOD.png")
    plt.close()



    l1 = ['ID']*len(safe.flatten())
    l2 = ['OOD']*len(risky.flatten())
    str_labels = ["ID" if i == 0 else 'OOD' for i in labels]
    df = pd.DataFrame({'label':str_labels, 'energy':examples})
    df.drop(df[(df['energy'] >100)].index, inplace=True)
    fig = sns.violinplot(x = df.label, y = df.energy)
    fig.axhline(31.06, label = "Threshold", color = 'brown')
    fig.set_xticks([0,1])
    plt.xlabel("Dataset Category", fontsize=15)
    plt.ylabel("Energy", fontsize=15)
    plt.legend(loc='upper center', fontsize = 15)
    fig = fig.get_figure()

    fig.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/RQ_figures_notebooks/RQ4/violinPlot.png", bbox_inches = "tight")


    _,_, fpr95 = get_curve(safe, risky)
    threshold = Find_Optimal_Cutoff(labels, examples)
    binaryExample = examples
    
    for i,e in enumerate(examples):
        if e > threshold:
            binaryExample[i] = 1
        else: 
            binaryExample[i] = 0
    
    tn, fp, fn, tp = sk.confusion_matrix(labels, binaryExample).ravel()
    accuracy  = (tp+tn)/(tp+tn+fp+fn)
    return AUROC, AUPR, fpr95, threshold[0], accuracy


def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, thr = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'thr' : pd.Series(thr, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    print("roc_t", roc_t)
    print("roc", roc)
    print("tpr", tpr[20127])
    print("fpr", fpr[20127])
    return list(roc_t['thr']) 

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
