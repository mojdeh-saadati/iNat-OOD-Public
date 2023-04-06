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
from sklearn.metrics import roc_curve, auc
import pandas as pd;
import ROCcurve
import random
import sys


sys.path.insert(1, '/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/RQ_figures_notebooks/RQ4/ROCcurve.py')
paths =["/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_49EBM.pt"]

def loadData_and_minSize(paths):
    minSize = -1;
    safeL = [];
    riskyL = []
    safe, risky = torch.load(paths[0]);  
    print("safe.type", type(safe))
    if not type(safe) is np.ndarray:
        safe, risky = safe.cpu().numpy(), risky.cpu().numpy()  

    if len(safe.shape) == 1:
        safe = safe.reshape((safe.shape[0],1))
    if len(risky.shape) == 1:
        risky = risky.reshape((risky.shape[0],1))
    safeL.append(safe), riskyL.append(risky);
    
    if(minSize == -1):
        minSize = min(safe.shape[0], risky.shape[0])
    else:
        minSize = min(minSize,safe.shape[0], risky.shape[0])

    safeIdx = random.sample(range(len(safe)), minSize)
    riskyIdx = random.sample(range(len(risky)), minSize)
    
    safe = safe[safeIdx];
    risky = risky[riskyIdx];

    return safe, risky 


totalResults = {"AUROC":[], "AUPR":[], "FPR95":[], "threshold":[] };
## load all lists:
safe, risky = loadData_and_minSize(paths)
print("safeList len", len(safe) )
accuracySum = 0;
thr = 0;
totalResults["AUROC"], totalResults["AUPR"], totalResults["FPR95"], totalResults["threshold"], totalResults["accuracy"] = ROCcurve.return_OOD_metrics_info(paths, safe, risky)

accuracySum += totalResults["accuracy"]
print("accuracySum", accuracySum)
print("accuracy total result", totalResults["accuracy"])

print( totalResults["threshold"])
thr = totalResults["threshold"]
print("threshold == ", thr)
