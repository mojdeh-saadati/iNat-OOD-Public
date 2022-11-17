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

import sys
sys.path.insert(1, '/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods')

import get_ood_metrics;


paths = ["/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results/model_0MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results/model_1MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results/model_10MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_results/model_49MSP.pt",
    
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results/model_0EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results/model_1EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results/model_10EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_results/model_49EBM.pt",

    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results/model_0MAH.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results/model_1MAH.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results/model_10MAH.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_results/model_49MAH.pt"]
chps = ["0","1","10","49"] *3
algs  = ["MSP"]*4 + ["EBM"]*4 + ["MAH"]*4 
import pandas as pd;
regnetRes = pd.DataFrame(columns = ["checkpoints","OODalg","AUROC", "AUPR", "FPR95"] );

for path,alg, chp in zip(paths,algs,chps):
    a = get_ood_metrics.return_OOD_metrics_info(path)
    regnetRes = regnetRes.append({'checkpoints' : chp, 'OODalg' : alg, "AUROC" : a[0], "AUPR": a[1],"FPR95":a[2]}, ignore_index = True)     
print("regnetRes === " , regnetRes) 
regnetRes.to_csv('/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/regnet32_result.csv') 

from matplotlib import pyplot as plt



plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MSP"], regnetRes.AUROC[regnetRes['OODalg'] == "MSP"] , label = "MSP");
plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "EBM"], regnetRes.AUROC[regnetRes['OODalg'] == "EBM"] , label = "EBM");
plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MAH"], regnetRes.AUROC[regnetRes['OODalg'] == "MAH"], label = "MAH");
plt.xlabel("AUROC")
plt.legend()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/regnetAUROC.png")
plt.close()

plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MSP"], regnetRes.AUPR[regnetRes['OODalg'] == "MSP"], label = "MSP");
plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "EBM"], regnetRes.AUPR[regnetRes['OODalg'] == "EBM"], label = "EBM");
plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MAH"], regnetRes.AUPR[regnetRes['OODalg'] == "MAH"], label = "MAH" );
plt.xlabel("AUPR")
plt.legend()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/regnetAUPR.png")
plt.close()

plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MSP"], regnetRes.FPR95[regnetRes['OODalg'] == "MSP"], label = "MSP");
plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "EBM"], regnetRes.FPR95[regnetRes['OODalg'] == "EBM"],  label = "EBM");
plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MAH"], regnetRes.FPR95[regnetRes['OODalg'] == "MAH"],  label = "MAH");
plt.xlabel("FPR95")
plt.legend()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/regnetFPR95.png")
plt.close()


"""

plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MSP"], regnetRes.AUROC[regnetRes['OODalg'] == "MSP"], label = "AUROC");
plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MSP"], regnetRes.AUPR[regnetRes['OODalg'] == "MSP"], label  = "AUPR");
plt.plot(regnetRes.checkpoints[regnetRes['OODalg'] == "MSP"], regnetRes.FPR95[regnetRes['OODalg'] == "MSP"], label = "FPR95");
plt.legend()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/regnetMSP.png")
plt.close()

"""