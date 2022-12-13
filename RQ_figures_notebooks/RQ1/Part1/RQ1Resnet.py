import numpy as np;
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
import pandas as pd;
from matplotlib import pyplot as plt
import random
import sys
sys.path.insert(1, '/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods')
random.seed(0)
import get_ood_metrics;

#################################################################
#   Loading the result vectors and ensure they are the same size
#   by random sampling subset of the larger one
#################################################################

def loadData_and_minSize(paths):
    minSize = -1;
    safeL = [];
    riskyL = []
    for path in paths:
        safe, risky = torch.load(path);  
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
    print("safeL type::::::::::::::::::::::::::::::", type(safeL))
    for i in range(len(safeL)):
        print("safe type:::::", type(safeL[i]),safeL[i].shape ,"    ", safeL[i][0:5])

        safeLS = random.sample(range(len(safeL[i])), minSize)
        riskyLS = random.sample(range(len(riskyL[i])), minSize)
        
        safeL[i] = safeL[i][safeLS];
        riskyL[i] = riskyL[i][riskyLS];

    return safeL, riskyL 

############################################################################
#        The list of all results vectors for MAH, MSP, EBM
############################################################################
paths = ["/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_0MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_1MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_10MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_20MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_30MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_40MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_49MSP.pt",
    
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_0EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_1EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_10EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_20EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_30EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_40EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_49EBM.pt",

    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_0MAH.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_1MAH.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_10MAH.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_20MAH.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_30MAH.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_40MAH.pt",    
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_49MAH.pt"]
chpsResnet = ["88.87","91.45","95.12","model_20", "model_30", "model_40","97.54"] *3

algs  = ["MSP"]*7 + ["EBM"]*7 + ["MAH"]*7 
result = pd.DataFrame(columns = ["checkpoints","OODalg","AUROC", "AUPR", "FPR95"] );
safeL, riskyL = loadData_and_minSize(paths)
for path,alg, chp, safe, risky in zip(paths,algs,chpsResnet, safeL, riskyL):
    a = get_ood_metrics.return_OOD_metrics_info(path, safe, risky)
    result = result.append({'checkpointsAcc' : chp, 'OODalg' : alg, "AUROC" : a[0], "AUPR": a[1],"FPR95":a[2]}, ignore_index = True)     
print("result === " , result) 
result.to_csv('/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/resnet50_result.csv') 



plt.plot(result.checkpointsAcc[result['OODalg'] == "MSP"], result.AUROC[result['OODalg'] == "MSP"] , label = "MSP");
plt.plot(result.checkpointsAcc[result['OODalg'] == "EBM"], result.AUROC[result['OODalg'] == "EBM"] , label = "EBM");
plt.plot(result.checkpointsAcc[result['OODalg'] == "MAH"], result.AUROC[result['OODalg'] == "MAH"], label = "MAH");
plt.xlabel("AUROC")
plt.legend()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/resnetAUROC.png")
plt.close()

plt.plot(result.checkpointsAcc[result['OODalg'] == "MSP"], result.AUPR[result['OODalg'] == "MSP"], label = "MSP");
plt.plot(result.checkpointsAcc[result['OODalg'] == "EBM"], result.AUPR[result['OODalg'] == "EBM"], label = "EBM");
plt.plot(result.checkpointsAcc[result['OODalg'] == "MAH"], result.AUPR[result['OODalg'] == "MAH"], label = "MAH" );
plt.xlabel("AUPR")
plt.legend()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/resnetAUPR.png")
plt.close()

plt.plot(result.checkpointsAcc[result['OODalg'] == "MSP"], result.FPR95[result['OODalg'] == "MSP"], label = "MSP");
plt.plot(result.checkpointsAcc[result['OODalg'] == "EBM"], result.FPR95[result['OODalg'] == "EBM"],  label = "EBM");
plt.plot(result.checkpointsAcc[result['OODalg'] == "MAH"], result.FPR95[result['OODalg'] == "MAH"],  label = "MAH");
plt.xlabel("FPR95")
plt.legend()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/resnetFPR95.png")
plt.close()


"""

plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MSP"], regnetRes.AUROC[regnetRes['OODalg'] == "MSP"], label = "AUROC");
plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MSP"], regnetRes.AUPR[regnetRes['OODalg'] == "MSP"], label  = "AUPR");
plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MSP"], regnetRes.FPR95[regnetRes['OODalg'] == "MSP"], label = "FPR95");
plt.legend()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/regnetMSP.png")
plt.close()

"""