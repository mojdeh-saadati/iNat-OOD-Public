import matplotlib.pyplot as plt
import numpy as np
import random;
import pandas as pd;
import sys
import torch;
import matplotlib;
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
sys.path.insert(1, '/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/OOD_methods')
import get_ood_metrics;


FONT_SIZE = 15 # font size for all text in the plots
font = {"size" : FONT_SIZE}  # font dict for rcParams
matplotlib.rc("font", **font)  # set font size for all text
matplotlib.rc("ytick", labelsize = FONT_SIZE)  # set font size for y-tick labels
matplotlib.rc("xtick", labelsize=FONT_SIZE)  # set font size for x-tick labels

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
    for i in range(len(safeL)):

        safeLS = random.sample(range(len(safeL[i])), minS)
        riskyLS = random.sample(range(len(riskyL[i])), minS)
        
        safeL[i] = safeL[i][safeLS];
        riskyL[i] = riskyL[i][riskyLS];

    return safeL, riskyL 

paths = [
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_b/MSP_results_nonInsecta2526/model_49MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub/MSP_results_nonInsecta2526/model_49MSP.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub_uniform/MSP_results_nonInsecta2526/model_49MSP.pt",

    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_b/EBM_results_nonInsecta2526/model_49EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub/EBM_results_nonInsecta2526/model_49EBM.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub_uniform/EBM_results_nonInsecta2526/model_49EBM.pt",

    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_b/MAH_results_nonInsecta2526/model_49MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub/MAH_results_nonInsecta2526/model_49MAH.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub_uniform/MAH_results_nonInsecta2526/model_49MAH.pt"

]



OODName = ["Balanced","Unbalanced", "Unbalanced_uniform"]*3;
OODalg = ["MSP"] *3 + ["EBM"]*3+ ["MAH"]*3;
result = pd.DataFrame(columns = ["AUROC"]);
safeL, riskyL = loadData_and_minSize(paths)
for path, safe, risky, alg, names in zip(paths,safeL, riskyL, OODalg, OODName):
    a = get_ood_metrics.return_OOD_metrics_info(path, safe,risky)
    result = result.append({'OODName' : names, 'OODalg' : alg, "AUROC" : a[0]}, ignore_index = True)     


# Figure 1 AUROC:
####################################################################################
labels = ['MSP', 'EBM', 'MAH']
["Balanced","Unbalanced"]
D1 = result.AUROC[result.OODName == "Balanced"]
D2 = result.AUROC[result.OODName == "Unbalanced"]
D3 = result.AUROC[result.OODName == "Unbalanced_uniform"]

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(1,1,figsize=(8,4.5))
rects1 = ax.bar(x - width, D1, width, label='Balanced', color = 'darkblue')
rects2 = ax.bar(x , D2, width, label='Unbalanced', color = 'salmon')
rects2 = ax.bar(x + width, D3, width, label='Unbalanced Uniform', color = 'brown')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUROC', fontsize = 15)
plt.xticks([r for r in range((3))],labels)
ax.set_xlabel('OOD Models',fontsize = 15)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
            fancybox=True, shadow=True)

fig.tight_layout()
plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/RQ_figures_notebooks/RQ2/RegNetY32AUROC_b_ub.pdf")
plt.close()

