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

def loadData_and_minSize(paths):
    minS = -1;
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
        
        if(minS == -1):
            minS = min(safe.shape[0], risky.shape[0])
        else:
            minS = min(minS,safe.shape[0], risky.shape[0])
    print("safeL type::::::::::::::::::::::::::::::", type(safeL))
    for i in range(len(safeL)):
        print("safe type:::::", type(safeL[i]),safeL[i].shape ,"    ", safeL[i][0:5])

        safeLS = random.sample(range(len(safeL[i])), minS)
        riskyLS = random.sample(range(len(riskyL[i])), minS)
        
        safeL[i] = safeL[i][safeLS];
        riskyL[i] = riskyL[i][riskyLS];

    return safeL, riskyL 

paths = ["/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_0MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_1MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_10MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_49MSP.pt",
    
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_0EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_1EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_10EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_49EBM.pt",

    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_0MAH.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_1MAH.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_10MAH.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_49MAH.pt"
    ]
chpsVGG = ["88.27","92.02","95.59","97.89"] *3
algs  = ["MSP"]*4 + ["EBM"]*4 + ["MAH"]*4 
vggRes = pd.DataFrame(columns = ["checkpoints","OODalg","AUROC", "AUPR", "FPR95"] );
safeL, riskyL = loadData_and_minSize(paths)
for path,alg, chp, safe, risky in zip(paths,algs,chpsVGG, safeL, riskyL):
    a = get_ood_metrics.return_OOD_metrics_info(path, safe, risky)
    vggRes = vggRes.append({'checkpoints' : chp, 'OODalg' : alg, "AUROC" : a[0], "AUPR": a[1],"FPR95":a[2]}, ignore_index = True)     
print("vggRes === " , vggRes) 
vggRes.to_csv('/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/VGG_result.csv') 



plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MSP"], vggRes.AUROC[vggRes['OODalg'] == "MSP"] , label = "MSP");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "EBM"], vggRes.AUROC[vggRes['OODalg'] == "EBM"] , label = "EBM");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MAH"], vggRes.AUROC[vggRes['OODalg'] == "MAH"], label = "MAH");
plt.xlabel("AUROC")
plt.legend()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/vggAUROC.png")
plt.close()

plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MSP"], vggRes.AUPR[vggRes['OODalg'] == "MSP"], label = "MSP");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "EBM"], vggRes.AUPR[vggRes['OODalg'] == "EBM"], label = "EBM");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MAH"], vggRes.AUPR[vggRes['OODalg'] == "MAH"], label = "MAH" );
plt.xlabel("AUPR")
plt.legend()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/vggAUPR.png")
plt.close()

plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MSP"], vggRes.FPR95[vggRes['OODalg'] == "MSP"], label = "MSP");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "EBM"], vggRes.FPR95[vggRes['OODalg'] == "EBM"],  label = "EBM");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MAH"], vggRes.FPR95[vggRes['OODalg'] == "MAH"],  label = "MAH");
plt.xlabel("FPR95")
plt.legend()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/vggFPR95.png")
plt.close()


"""

plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MSP"], vggRes.AUROC[vggRes['OODalg'] == "MSP"], label = "AUROC");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MSP"], vggRes.AUPR[vggRes['OODalg'] == "MSP"], label  = "AUPR");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MSP"], vggRes.FPR95[vggRes['OODalg'] == "MSP"], label = "FPR95");
plt.legend()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/regnetMSP.png")
plt.close()

"""