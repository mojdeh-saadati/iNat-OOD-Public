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
paths = ["/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_0MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_1MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_2MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_3MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_4MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_5MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_6MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_7MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_8MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_9MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_10MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_20MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_30MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_40MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MSP_results_nonInsecta2526/model_49MSP.pt",
    
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_0EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_1EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_2EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_3EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_4EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_5EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_6EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_7EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_8EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_9EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_10EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_20EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_30EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_40EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/EBM_results_nonInsecta2526/model_49EBM.pt",

    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_0MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_1MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_2MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_3MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_4MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_5MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_6MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_7MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_8MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_9MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_10MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_20MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_30MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_40MAH.pt",    
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/MAH_results_nonInsecta2526/model_49MAH.pt"]


chpsResnet = [88.87,91.45,92.17,93.05,93.51,94.18,94.46,94.58,94.47,94.64,95.12,96.09, 97.15, 97.48,97.54] *3

algs  = ["MSP"]*15 + ["EBM"]*15 + ["MAH"]*15 
resnetRes = pd.DataFrame(columns = ["checkpoints","OODalg","AUROC", "AUPR", "FPR95"] );
safeL, riskyL = loadData_and_minSize(paths)
for path,alg, chp, safe, risky in zip(paths,algs,chpsResnet, safeL, riskyL):
    a = get_ood_metrics.return_OOD_metrics_info(path, safe, risky)
    resnetRes = resnetRes.append({'checkpointsAcc' : chp, 'OODalg' : alg, "AUROC" : a[0], "AUPR": a[1],"FPR95":a[2]}, ignore_index = True)     
print("result === " , resnetRes) 

resnetRes.to_csv('/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/resnet50_result.csv') 
plt.plot(resnetRes[resnetRes.OODalg =='MSP'].checkpointsAcc, resnetRes[resnetRes.OODalg =='MSP'].AUROC, 'o', color='darkblue')
m1, b1 = np.polyfit(resnetRes[resnetRes.OODalg =='MSP'].checkpointsAcc, resnetRes[resnetRes.OODalg =='MSP'].AUROC , 1)
plt.plot(resnetRes[resnetRes.OODalg =='MSP'].checkpointsAcc, m1*resnetRes[resnetRes.OODalg =='MSP'].checkpointsAcc+b1, color='darkblue')
plt.plot(resnetRes[resnetRes.OODalg =='MAH'].checkpointsAcc, resnetRes[resnetRes.OODalg =='MAH'].AUROC, 'o', color='orange')
m2, b2 = np.polyfit(resnetRes[resnetRes.OODalg =='MAH'].checkpointsAcc, resnetRes[resnetRes.OODalg =='MAH'].AUROC , 1)
plt.plot(resnetRes[resnetRes.OODalg =='MAH'].checkpointsAcc, m2*resnetRes[resnetRes.OODalg =='MAH'].checkpointsAcc+b2, color='orange')
plt.plot(resnetRes[resnetRes.OODalg =='EBM'].checkpointsAcc, resnetRes[resnetRes.OODalg =='EBM'].AUROC, 'o', color='brown')
m3, b3 = np.polyfit(resnetRes[resnetRes.OODalg =='EBM'].checkpointsAcc, resnetRes[resnetRes.OODalg =='EBM'].AUROC , 1)
plt.plot(resnetRes[resnetRes.OODalg =='EBM'].checkpointsAcc, m3*resnetRes[resnetRes.OODalg =='EBM'].checkpointsAcc+b3, color='brown')


plt.ylabel("AUROC")
plt.xlabel("Accuracies")
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/ResNet50AUROC.png")
plt.close()



"""
plt.plot(result.checkpointsAcc[result['OODalg'] == "MSP"], result.AUROC[result['OODalg'] == "MSP"] , label = "MSP");
plt.plot(result.checkpointsAcc[result['OODalg'] == "EBM"], result.AUROC[result['OODalg'] == "EBM"] , label = "EBM");
plt.plot(result.checkpointsAcc[result['OODalg'] == "MAH"], result.AUROC[result['OODalg'] == "MAH"], label = "MAH");
plt.xlabel("AUROC")
plt.legend()
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/resnetAUROC.png")
plt.close()

plt.plot(result.checkpointsAcc[result['OODalg'] == "MSP"], result.AUPR[result['OODalg'] == "MSP"], label = "MSP");
plt.plot(result.checkpointsAcc[result['OODalg'] == "EBM"], result.AUPR[result['OODalg'] == "EBM"], label = "EBM");
plt.plot(result.checkpointsAcc[result['OODalg'] == "MAH"], result.AUPR[result['OODalg'] == "MAH"], label = "MAH" );
plt.xlabel("AUPR")
plt.legend()
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/resnetAUPR.png")
plt.close()

plt.plot(result.checkpointsAcc[result['OODalg'] == "MSP"], result.FPR95[result['OODalg'] == "MSP"], label = "MSP");
plt.plot(result.checkpointsAcc[result['OODalg'] == "EBM"], result.FPR95[result['OODalg'] == "EBM"],  label = "EBM");
plt.plot(result.checkpointsAcc[result['OODalg'] == "MAH"], result.FPR95[result['OODalg'] == "MAH"],  label = "MAH");
plt.xlabel("FPR95")
plt.legend()
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/resnetFPR95.png")
plt.close()


plt.plot(resnetRes.checkpoints[resnetRes['OODalg'] == "MSP"], resnetRes.AUROC[resnetRes['OODalg'] == "MSP"], label = "AUROC");
plt.plot(resnetRes.checkpoints[resnetRes['OODalg'] == "MSP"], resnetRes.AUPR[resnetRes['OODalg'] == "MSP"], label  = "AUPR");
plt.plot(resnetRes.checkpoints[resnetRes['OODalg'] == "MSP"], resnetRes.FPR95[resnetRes['OODalg'] == "MSP"], label = "FPR95");
plt.legend()
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Resnet50-142/resnetMSP.png")
plt.close()

"""