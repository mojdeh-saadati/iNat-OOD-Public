import matplotlib.pyplot as plt
import numpy as np
import random;
import pandas as pd;
import sys
import torch;
sys.path.insert(1, '/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods')

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

paths = ["/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_allOOD/model_49MSPCorrect_and_NoMask_facemask_recog_datasets.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_allOOD/model_49MSP_imagenet_2012_5000-NoInsects.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_allOOD/model_49MSP_nonInsecta.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_allOOD/model_49MSP_OODInsects.pt",
    
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_allOOD/model_49EBMCorrect_and_NoMask_facemask_recog_datasets.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_allOOD/model_49EBM_imagenet_2012_5000-NoInsects.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_allOOD/model_49EBM_nonInsecta.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_allOOD/model_49EBM_OODInsects.pt",

    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_allOOD/model_49MAHCorrect_and_NoMask_facemask_recog_datasets.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_allOOD/model_49MAH_imagenet_2012_5000-NoInsects.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_allOOD/model_49MAH_nonInsecta.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_allOOD/model_49MAH_OODInsects.pt"]



OODName = ["HumanFace","Imagenet", "NonInsecta", "OODInsects"]*4;
OODalg = ["MSP"] *4 + ["EBM"]*4+ ["MAH"]*4;
result = pd.DataFrame(columns = ["OODName","OODalg","AUROC", "AUPR", "FPR95"] );
safeL, riskyL = loadData_and_minSize(paths)
for path, safe, risky, alg, names in zip(paths,safeL, riskyL, OODalg, OODName):
    a = get_ood_metrics.return_OOD_metrics_info(path, safe,risky)
    result = result.append({'OODName' : names, 'OODalg' : alg, "AUROC" : a[0], "AUPR": a[1],"FPR95":a[2]}, ignore_index = True)     
print("result === " , result) 
result.to_csv('/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/OODRegnet32_result.csv') 



# Figure 1 AUROC:
####################################################################################
labels = ['MSP', 'EBM', 'MAH']
["HumanFace","Imagenet", "NonInsecta", "OODInsects"]
D1 = result.AUROC[result.OODName == "HumanFace"]
D2 = result.AUROC[result.OODName == "Imagenet"]
D3 = result.AUROC[result.OODName == "NonInsecta"]
D4 = result.AUROC[result.OODName == "OODInsects"]

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x -width - (width/2), D1, width, label='Human')
rects2 = ax.bar(x - width/2, D2, width, label='Imagenet')
rects3 = ax.bar(x + width/2, D3, width, label='nonInsecta')
rects4 = ax.bar(x + width + width/2, D4, width, label='OODInsects')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUROC')
ax.set_title('AUROC for out-of-distribution datasets')
plt.xticks([r for r in range((3))],labels)
ax.legend()

fig.tight_layout()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/regnet32AUROC_allOOD.png")
plt.close()


# Figure 2 AUPR:
####################################################################################

labels = ['MSP', 'EBM', 'MAH']
["HumanFace","Imagenet", "NonInsecta", "OODInsects"]
D1 = result.AUPR[result.OODName == "HumanFace"]
D2 = result.AUPR[result.OODName == "Imagenet"]
D3 = result.AUPR[result.OODName == "NonInsecta"]
D4 = result.AUPR[result.OODName == "OODInsects"]

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x -width - (width/2), D1, width, label='Human')
rects2 = ax.bar(x - width/2, D2, width, label='Imagenet')
rects3 = ax.bar(x + width/2, D3, width, label='nonInsecta')
rects4 = ax.bar(x + width + width/2, D4, width, label='OODInsects')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUPR')
ax.set_title('AUPR for out-of-distribution datasets')
ax.legend()
plt.xticks([r for r in range((3))],labels)
fig.tight_layout()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/regnet32AUPR_allOOD.png")
plt.close()

# Figure 3 FPR95:
####################################################################################
labels = ['MSP', 'EBM', 'MAH']
["HumanFace","Imagenet", "NonInsecta", "OODInsects"]
D1 = result.FPR95[result.OODName == "HumanFace"]
D2 = result.FPR95[result.OODName == "Imagenet"]
D3 = result.FPR95[result.OODName == "NonInsecta"]
D4 = result.FPR95[result.OODName == "OODInsects"]

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x -width - (width/2), D1, width, label='Human')
rects2 = ax.bar(x - width/2, D2, width, label='Imagenet')
rects3 = ax.bar(x + width/2, D3, width, label='nonInsecta')
rects4 = ax.bar(x + width + width/2, D4, width, label='OODInsects')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FPR95')
ax.set_title('FPR95 for out-of-distribution datasets')

plt.xticks([r for r in range((3))],labels)
ax.legend()
fig.tight_layout()

plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/regnet32FPR95_allOOD.png")
plt.close()
