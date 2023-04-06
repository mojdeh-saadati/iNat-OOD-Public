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
sys.path.insert(1, '/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods')
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
paths = ["/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_0MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_1MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_2MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_3MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_4MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_5MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_6MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_7MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_8MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_9MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_10MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_20MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_30MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_40MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results_nonInsecta2526/model_49MSP.pt",
    
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_0EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_1EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_2EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_3EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_4EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_5EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_6EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_7EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_8EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_9EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_10EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_20EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_30EBM.pt",    
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_40EBM.pt",    
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results_nonInsecta2526/model_49EBM.pt",

    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_0MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_1MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_2MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_3MAH.pt",   
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_4MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_5MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_6MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_7MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_8MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_9MAH.pt",                             
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_10MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_20MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_30MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_40MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results_nonInsecta2526/model_49MAH.pt"]

chpsRegnet = [88.27,92.02,93.45,93.83,94.64,94.93,95.40, 93.79, 95.57,95.82,95.59,96.79, 97.51,97.82,97.89] *3
algs  = ["MSP"]*15 + ["EBM"]*15 + ["MAH"]*15
regnetRes = pd.DataFrame(columns = ["checkpointsAcc","OODalg","AUROC", "AUPR", "FPR95"] );
safeL, riskyL = loadData_and_minSize(paths)
for path,alg, chp, safe, risky in zip(paths,algs,chpsRegnet, safeL, riskyL):
    a = get_ood_metrics.return_OOD_metrics_info(path, safe, risky)
    regnetRes = regnetRes.append({'checkpointsAcc' : chp, 'OODalg' : alg, "AUROC" : a[0], "AUPR": a[1],"FPR95":a[2]}, ignore_index = True)     
print("regnetRes === " , regnetRes) 
regnetRes.to_csv('/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/regnet32_result.csv') 
plt.plot(regnetRes[regnetRes.OODalg =='MSP'].checkpointsAcc, regnetRes[regnetRes.OODalg =='MSP'].AUROC, 'o', color='darkblue')
m1, b1 = np.polyfit(regnetRes[regnetRes.OODalg =='MSP'].checkpointsAcc, regnetRes[regnetRes.OODalg =='MSP'].AUROC , 1)
plt.plot(regnetRes[regnetRes.OODalg =='MSP'].checkpointsAcc, m1*regnetRes[regnetRes.OODalg =='MSP'].checkpointsAcc+b1, color='darkblue')
plt.plot(regnetRes[regnetRes.OODalg =='MAH'].checkpointsAcc, regnetRes[regnetRes.OODalg =='MAH'].AUROC, 'o', color='orange')
m2, b2 = np.polyfit(regnetRes[regnetRes.OODalg =='MAH'].checkpointsAcc, regnetRes[regnetRes.OODalg =='MAH'].AUROC , 1)
plt.plot(regnetRes[regnetRes.OODalg =='MAH'].checkpointsAcc, m2*regnetRes[regnetRes.OODalg =='MAH'].checkpointsAcc+b2, color='orange')
plt.plot(regnetRes[regnetRes.OODalg =='EBM'].checkpointsAcc, regnetRes[regnetRes.OODalg =='EBM'].AUROC, 'o', color='brown')
m3, b3 = np.polyfit(regnetRes[regnetRes.OODalg =='EBM'].checkpointsAcc, regnetRes[regnetRes.OODalg =='EBM'].AUROC , 1)
plt.plot(regnetRes[regnetRes.OODalg =='EBM'].checkpointsAcc, m3*regnetRes[regnetRes.OODalg =='EBM'].checkpointsAcc+b3, color='brown')


plt.ylabel("AUROC")
plt.xlabel("Accuracies")
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/RegNet32AUROC.png")
plt.close()

"""
plt.plot(regnetRes.checkpointsAcc[regnetRes['OODalg'] == "MSP"], regnetRes.AUROC[regnetRes['OODalg'] == "MSP"] , label = "MSP");
plt.plot(regnetRes.checkpointsAcc[regnetRes['OODalg'] == "EBM"], regnetRes.AUROC[regnetRes['OODalg'] == "EBM"] , label = "EBM");
plt.plot(regnetRes.checkpointsAcc[regnetRes['OODalg'] == "MAH"], regnetRes.AUROC[regnetRes['OODalg'] == "MAH"] , label = "MAH");

plt.xlabel("AUROC")
plt.legend()
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/regnetAUROC.png")
plt.close()

plt.plot(regnetRes.checkpointsAcc[regnetRes['OODalg'] == "MSP"], regnetRes.AUPR[regnetRes['OODalg'] == "MSP"], label = "MSP");
plt.plot(regnetRes.checkpointsAcc[regnetRes['OODalg'] == "EBM"], regnetRes.AUPR[regnetRes['OODalg'] == "EBM"], label = "EBM");
plt.plot(regnetRes.checkpointsAcc[regnetRes['OODalg'] == "MAH"], regnetRes.AUPR[regnetRes['OODalg'] == "MAH"], label = "MAH" );
plt.xlabel("AUPR")
plt.legend()
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/regnetAUPR.png")
plt.close()

plt.plot(regnetRes.checkpointsAcc[regnetRes['OODalg'] == "MSP"], regnetRes.FPR95[regnetRes['OODalg'] == "MSP"], label = "MSP");
plt.plot(regnetRes.checkpointsAcc[regnetRes['OODalg'] == "EBM"], regnetRes.FPR95[regnetRes['OODalg'] == "EBM"],  label = "EBM");
plt.plot(regnetRes.checkpointsAcc[regnetRes['OODalg'] == "MAH"], regnetRes.FPR95[regnetRes['OODalg'] == "MAH"],  label = "MAH");
plt.xlabel("FPR95")
plt.legend()
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/regnetFPR95.png")
plt.close()

plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MSP"], regnetRes.AUROC[regnetRes['OODalg'] == "MSP"], label = "AUROC");
plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MSP"], regnetRes.AUPR[regnetRes['OODalg'] == "MSP"], label  = "AUPR");
plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MSP"], regnetRes.FPR95[regnetRes['OODalg'] == "MSP"], label = "FPR95");
plt.legend()
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/regnetMSP.png")
plt.close()
"""