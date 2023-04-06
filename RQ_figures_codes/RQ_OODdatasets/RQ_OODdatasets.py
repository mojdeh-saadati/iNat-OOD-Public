import matplotlib.pyplot as plt
import numpy as np
import random;
import pandas as pd;
import sys
import torch;
import matplotlib;
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

paths = ["/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_allOOD/model_49MSPCorrect_and_NoMask_facemask_recog_datasets.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_allOOD/model_49MSP_imagenet_2012_5000-NoInsects.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_allOOD/model_49MSP_nonInsecta.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MSP_allOOD/model_49MSP_OODInsects.pt",
    
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_allOOD/model_49EBMCorrect_and_NoMask_facemask_recog_datasets.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_allOOD/model_49EBM_imagenet_2012_5000-NoInsects.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_allOOD/model_49EBM_nonInsecta.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/EBM_allOOD/model_49EBM_OODInsects.pt",

    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_allOOD/model_49MAHCorrect_and_NoMask_facemask_recog_datasets.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_allOOD/model_49MAH_imagenet_2012_5000-NoInsects.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_allOOD/model_49MAH_nonInsecta.pt",
    "/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/MAH_allOOD/model_49MAH_OODInsects.pt"]



OODName = ["HumanFace","Imagenet", "NonInsecta", "OODInsects"]*4;
OODalg = ["MSP"] *4 + ["EBM"]*4+ ["MAH"]*4;
result = pd.DataFrame(columns = ["OODName","OODalg","AUROC", "AUPR", "FPR95"] );
safeL, riskyL = loadData_and_minSize(paths)

for path, safe, risky, alg, names in zip(paths,safeL, riskyL, OODalg, OODName):
    a = get_ood_metrics.return_OOD_metrics_info(path, safe,risky)
    result = result.append({'OODName' : names, 'OODalg' : alg, "AUROC" : a[0], "AUPR": a[1],"FPR95":a[2]}, ignore_index = True)     
print("result === " , result) 
result.to_csv('/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32-142/OODRegnet32_result.csv') 



# Figure 1 AUROC:
####################################################################################

labels = ['MSP', 'EBM', 'MAH']
#["HumanFace","Imagenet", "NonInsecta", "OODInsects"]
D1 = result.AUROC[result.OODName == "HumanFace"]
D2 = result.AUROC[result.OODName == "Imagenet"]
D3 = result.AUROC[result.OODName == "NonInsecta"]
D4 = result.AUROC[result.OODName == "OODInsects"]

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig,ax = plt.subplots(1,2,figsize=(12,4.5))
rects1 = ax[0].bar(x -width - (width/2), D1, width, label='HumanFace', color = 'lightskyblue')
rects2 = ax[0].bar(x - width/2, D2, width, label='Imagenet',color ='darkblue' )
rects3 = ax[0].bar(x + width/2, D3, width, label='nonInsecta', color = 'salmon')
rects4 = ax[0].bar(x + width + width/2, D4, width, label='OODInsects' , color = 'brown')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('AUROC', fontsize = 15)
ax[0].set_xlabel('OOD Models', fontsize = 15)

# Figure 2 FPR95:
####################################################################################

D1 = result.FPR95[result.OODName == "HumanFace"]
D2 = result.FPR95[result.OODName == "Imagenet"]
D3 = result.FPR95[result.OODName == "NonInsecta"]
D4 = result.FPR95[result.OODName == "OODInsects"]

rects1 = ax[1].bar(x -width - (width/2), D1, width, color = 'lightskyblue') # label= "HumanFace"
rects2 = ax[1].bar(x - width/2, D2, width , color ='darkblue') # label='Imagenet'
rects3 = ax[1].bar(x + width/2, D3, width ,color = 'salmon') # label='nonInsecta'
rects4 = ax[1].bar(x + width + width/2, D4, width,color = 'brown') #label='OODInsects'


print("rects1 ==", D1)
# Add some text for labels, title and custom x-axis tick labels, etc.

ax[1].set_ylabel('FPR95', fontsize = 15)
ax[1].set_xlabel('OOD Models', fontsize = 15)
plt.setp(ax, xticks=[r for r in range((3))], xticklabels = labels)
fig.tight_layout()
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.00),
            fancybox=True, shadow=True, ncol=4)


plt.savefig("/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/RQ_figures_notebooks/RQ3/RQ3.pdf", bbox_inches = 'tight')
plt.close()
