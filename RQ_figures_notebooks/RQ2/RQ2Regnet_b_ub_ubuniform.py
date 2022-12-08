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

paths = [
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_b/MSP_results_nonInsecta2526/model_49MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub/MSP_results_nonInsecta2526/model_49MSP.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub_uniform/MSP_results_nonInsecta2526/model_49MSP.pt",

    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_b/EBM_results_nonInsecta2526/model_49EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub/EBM_results_nonInsecta2526/model_49EBM.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub_uniform/EBM_results_nonInsecta2526/model_49EBM.pt",

    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_b/MAH_results_nonInsecta2526/model_49MAH.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub/MAH_results_nonInsecta2526/model_49MAH.pt",
    "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub_uniform/MAH_results_nonInsecta2526/model_49MAH.pt"

]



OODName = ["Balanced","Unbalanced", "Unbalanced_uniform"]*3;
OODalg = ["MSP"] *3 + ["EBM"]*3+ ["MAH"]*3;
result = pd.DataFrame(columns = ["AUROC"]);
safeL, riskyL = loadData_and_minSize(paths)
for path, safe, risky, alg, names in zip(paths,safeL, riskyL, OODalg, OODName):
    a = get_ood_metrics.return_OOD_metrics_info(path, safe,risky)
    result = result.append({'OODName' : names, 'OODalg' : alg, "AUROC" : a[0]}, ignore_index = True)     


print("result === " , result) 

# Figure 1 AUROC:
####################################################################################
labels = ['MSP', 'EBM', 'MAH']
["Balanced","Unbalanced"]
D1 = result.AUROC[result.OODName == "Balanced"]
D2 = result.AUROC[result.OODName == "Unbalanced"]
D3 = result.AUROC[result.OODName == "Unbalanced_uniform"]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, D1, width, label='Balanced')
rects2 = ax.bar(x , D2, width, label='Unbalanced')
rects2 = ax.bar(x + width/2, D3, width, label='Unbalanced_uniform')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUROC')
ax.set_title('AUROC for out-of-distribution datasets')
plt.xticks([r for r in range((3))],labels)
ax.legend()

fig.tight_layout()
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_b/regnet32AUROC_b_ub.png")
plt.close()

