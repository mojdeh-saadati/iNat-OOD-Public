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

paths = ["/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_0MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_1MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_2MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_3MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_4MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_5MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_6MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_7MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_8MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_9MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_10MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_20MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_30MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_40MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MSP_results_nonInsecta2526/model_49MSP.pt",
    
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_0EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_1EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_2EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_3EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_4EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_5EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_6EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_7EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_8EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_9EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_10EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_20EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_30EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_40EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/EBM_results_nonInsecta2526/model_49EBM.pt",

    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_0MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_1MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_2MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_3MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_4MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_5MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_6MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_7MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_8MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_9MAH.pt",   
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_10MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_20MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_30MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_40MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/MAH_results_nonInsecta2526/model_49MAH.pt"]

chpsVGG = [80.09,83.93,85.93,87.19,87.93,88.40,89.41,89.70,89.60,90.23,90.57,91.78,93.88,94.50,94.63] *3
algs  = ["MSP"]*15 + ["EBM"]*15 + ["MAH"]*15 
VGGRes = pd.DataFrame(columns = ["checkpointsAcc","OODalg","AUROC", "AUPR", "FPR95"] );
safeL, riskyL = loadData_and_minSize(paths)
for path,alg, chp, safe, risky in zip(paths,algs,chpsVGG, safeL, riskyL):
    a = get_ood_metrics.return_OOD_metrics_info(path, safe, risky)
    VGGRes = VGGRes.append({'checkpointsAcc' : chp, 'OODalg' : alg, "AUROC" : a[0], "AUPR": a[1],"FPR95":a[2]}, ignore_index = True)     
print("VGGRes === " , VGGRes) 

VGGRes.to_csv('/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/VGG11_result.csv') 
plt.plot(VGGRes[VGGRes.OODalg =='MSP'].checkpointsAcc, VGGRes[VGGRes.OODalg =='MSP'].AUROC, 'o', color='darkblue')
m1, b1 = np.polyfit(VGGRes[VGGRes.OODalg =='MSP'].checkpointsAcc, VGGRes[VGGRes.OODalg =='MSP'].AUROC , 1)
plt.plot(VGGRes[VGGRes.OODalg =='MSP'].checkpointsAcc, m1*VGGRes[VGGRes.OODalg =='MSP'].checkpointsAcc+b1, color='darkblue')
plt.plot(VGGRes[VGGRes.OODalg =='MAH'].checkpointsAcc, VGGRes[VGGRes.OODalg =='MAH'].AUROC, 'o', color='orange')
m2, b2 = np.polyfit(VGGRes[VGGRes.OODalg =='MAH'].checkpointsAcc, VGGRes[VGGRes.OODalg =='MAH'].AUROC , 1)
plt.plot(VGGRes[VGGRes.OODalg =='MAH'].checkpointsAcc, m2*VGGRes[VGGRes.OODalg =='MAH'].checkpointsAcc+b2, color='orange')
plt.plot(VGGRes[VGGRes.OODalg =='EBM'].checkpointsAcc, VGGRes[VGGRes.OODalg =='EBM'].AUROC, 'o', color='brown')
m3, b3 = np.polyfit(VGGRes[VGGRes.OODalg =='EBM'].checkpointsAcc, VGGRes[VGGRes.OODalg =='EBM'].AUROC , 1)
plt.plot(VGGRes[VGGRes.OODalg =='EBM'].checkpointsAcc, m3*VGGRes[VGGRes.OODalg =='EBM'].checkpointsAcc+b3, color='brown')

plt.ylabel("AUROC")
plt.xlabel("Accuracies")
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/VGG11AUROC.png")
plt.close()




"""

plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MSP"], vggRes.AUROC[vggRes['OODalg'] == "MSP"] , label = "MSP");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "EBM"], vggRes.AUROC[vggRes['OODalg'] == "EBM"] , label = "EBM");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MAH"], vggRes.AUROC[vggRes['OODalg'] == "MAH"], label = "MAH");
plt.xlabel("AUROC")
plt.legend()
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/vggAUROC.png")
plt.close()

plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MSP"], vggRes.AUPR[vggRes['OODalg'] == "MSP"], label = "MSP");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "EBM"], vggRes.AUPR[vggRes['OODalg'] == "EBM"], label = "EBM");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MAH"], vggRes.AUPR[vggRes['OODalg'] == "MAH"], label = "MAH" );
plt.xlabel("AUPR")
plt.legend()
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/vggAUPR.png")
plt.close()

plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MSP"], vggRes.FPR95[vggRes['OODalg'] == "MSP"], label = "MSP");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "EBM"], vggRes.FPR95[vggRes['OODalg'] == "EBM"],  label = "EBM");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MAH"], vggRes.FPR95[vggRes['OODalg'] == "MAH"],  label = "MAH");
plt.xlabel("FPR95")
plt.legend()
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/vggFPR95.png")
plt.close()


plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MSP"], vggRes.AUROC[vggRes['OODalg'] == "MSP"], label = "AUROC");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MSP"], vggRes.AUPR[vggRes['OODalg'] == "MSP"], label  = "AUPR");
plt.plot(vggRes.checkpoints[vggRes['OODalg'] == "MSP"], vggRes.FPR95[vggRes['OODalg'] == "MSP"], label = "FPR95");
plt.legend()
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/VGG-142/regnetMSP.png")
plt.close()

"""