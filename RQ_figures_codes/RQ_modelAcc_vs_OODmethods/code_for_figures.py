import pandas as pd;
import matplotlib.pyplot as plt;
import matplotlib;
import numpy as np;
import seaborn as sns
plt.style.use('default')
import matplotlib.font_manager;  # for setting font size
regnetRes = pd.DataFrame()
resnetRes = pd.DataFrame()
VGGRes = pd.DataFrame()

FONT_SIZE = 17 # font size for all text in the plots
font = {"size" : FONT_SIZE}  # font dict for rcParams
matplotlib.rc("font", **font)  # set font size for all text
matplotlib.rc("ytick", labelsize = FONT_SIZE)  # set font size for y-tick labels
matplotlib.rc("xtick", labelsize=FONT_SIZE)  # set font size for x-tick labels


fig,axs = plt.subplots(1,3,sharey=True,figsize=(16,4))

resnetRes = pd.read_csv('/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/RQ_figures_codes/RQ_modelAcc_vs_OODmethods/resnet50_result.csv') 
axs[0].plot(resnetRes[resnetRes.OODalg =='MSP'].checkpointsAcc, resnetRes[resnetRes.OODalg =='MSP'].AUROC, 'o', color='darkblue')
m1, b1 = np.polyfit(resnetRes[resnetRes.OODalg =='MSP'].checkpointsAcc, resnetRes[resnetRes.OODalg =='MSP'].AUROC , 1)
axs[0].plot(resnetRes[resnetRes.OODalg =='MSP'].checkpointsAcc, m1*resnetRes[resnetRes.OODalg =='MSP'].checkpointsAcc+b1, linestyle='dashed', color='darkblue')
axs[0].plot(resnetRes[resnetRes.OODalg =='MAH'].checkpointsAcc, resnetRes[resnetRes.OODalg =='MAH'].AUROC, '^', color='orange')
m2, b2 = np.polyfit(resnetRes[resnetRes.OODalg =='MAH'].checkpointsAcc, resnetRes[resnetRes.OODalg =='MAH'].AUROC , 1)
axs[0].plot(resnetRes[resnetRes.OODalg =='MAH'].checkpointsAcc, m2*resnetRes[resnetRes.OODalg =='MAH'].checkpointsAcc+b2,linestyle='dashed', color='orange')
axs[0].plot(resnetRes[resnetRes.OODalg =='EBM'].checkpointsAcc, resnetRes[resnetRes.OODalg =='EBM'].AUROC, 'D', color='brown')
m3, b3 = np.polyfit(resnetRes[resnetRes.OODalg =='EBM'].checkpointsAcc, resnetRes[resnetRes.OODalg =='EBM'].AUROC , 1)
axs[0].plot(resnetRes[resnetRes.OODalg =='EBM'].checkpointsAcc, m3*resnetRes[resnetRes.OODalg =='EBM'].checkpointsAcc+b3, linestyle='dashed', color='brown')
axs[0].set_ylabel("AUROC")
axs[0].set_xlabel("Accuracies")
axs[0].set_title('ResNet50 23mp') 


VGGRes = pd.read_csv('/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/RQ_figures_codes/RQ_modelAcc_vs_OODmethods/VGG11_result.csv') 
axs[1].plot(VGGRes[VGGRes.OODalg =='MSP'].checkpointsAcc, VGGRes[VGGRes.OODalg =='MSP'].AUROC, 'o', color='darkblue')
m1, b1 = np.polyfit(VGGRes[VGGRes.OODalg =='MSP'].checkpointsAcc, VGGRes[VGGRes.OODalg =='MSP'].AUROC , 1)
axs[1].plot(VGGRes[VGGRes.OODalg =='MSP'].checkpointsAcc, m1*VGGRes[VGGRes.OODalg =='MSP'].checkpointsAcc+b1, label='MSP', linestyle='dashed',color='darkblue')
axs[1].plot(VGGRes[VGGRes.OODalg =='MAH'].checkpointsAcc, VGGRes[VGGRes.OODalg =='MAH'].AUROC, '^',color='orange')
m2, b2 = np.polyfit(VGGRes[VGGRes.OODalg =='MAH'].checkpointsAcc, VGGRes[VGGRes.OODalg =='MAH'].AUROC , 1)
axs[1].plot(VGGRes[VGGRes.OODalg =='MAH'].checkpointsAcc, m2*VGGRes[VGGRes.OODalg =='MAH'].checkpointsAcc+b2, label='MAH' ,linestyle='dashed', color='orange')
axs[1].plot(VGGRes[VGGRes.OODalg =='EBM'].checkpointsAcc, VGGRes[VGGRes.OODalg =='EBM'].AUROC, 'D',color='brown')
m3, b3 = np.polyfit(VGGRes[VGGRes.OODalg =='EBM'].checkpointsAcc, VGGRes[VGGRes.OODalg =='EBM'].AUROC , 1)
axs[1].plot(VGGRes[VGGRes.OODalg =='EBM'].checkpointsAcc, m3*VGGRes[VGGRes.OODalg =='EBM'].checkpointsAcc+b3, label='EBM',linestyle='dashed', color='brown')
axs[1].set_xlabel("Accuracies")
axs[1].set_title('VGG11 133mp') 




regnetRes = pd.read_csv('/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/RQ_figures_codes/RQ_modelAcc_vs_OODmethods/regnet32_result.csv') 
axs[2].plot(regnetRes[regnetRes.OODalg =='MSP'].checkpointsAcc, regnetRes[regnetRes.OODalg =='MSP'].AUROC, 'o', color='darkblue')
m1, b1 = np.polyfit(regnetRes[regnetRes.OODalg =='MSP'].checkpointsAcc, regnetRes[regnetRes.OODalg =='MSP'].AUROC , 1)
axs[2].plot(regnetRes[regnetRes.OODalg =='MSP'].checkpointsAcc, m1*regnetRes[regnetRes.OODalg =='MSP'].checkpointsAcc+b1,linestyle='dashed' ,color='darkblue')
axs[2].plot(regnetRes[regnetRes.OODalg =='MAH'].checkpointsAcc, regnetRes[regnetRes.OODalg =='MAH'].AUROC, '^', color='orange')
m2, b2 = np.polyfit(regnetRes[regnetRes.OODalg =='MAH'].checkpointsAcc, regnetRes[regnetRes.OODalg =='MAH'].AUROC , 1)
axs[2].plot(regnetRes[regnetRes.OODalg =='MAH'].checkpointsAcc, m2*regnetRes[regnetRes.OODalg =='MAH'].checkpointsAcc+b2,linestyle='dashed', color='orange')
axs[2].plot(regnetRes[regnetRes.OODalg =='EBM'].checkpointsAcc, regnetRes[regnetRes.OODalg =='EBM'].AUROC, 'D', color='brown')
m3, b3 = np.polyfit(regnetRes[regnetRes.OODalg =='EBM'].checkpointsAcc, regnetRes[regnetRes.OODalg =='EBM'].AUROC , 1)
axs[2].plot(regnetRes[regnetRes.OODalg =='EBM'].checkpointsAcc, m3*regnetRes[regnetRes.OODalg =='EBM'].checkpointsAcc+b3,linestyle='dashed',color='brown')
axs[2].set_xlabel("Accuracies")
axs[2].set_title('RegNetY32 145mp') 


fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)
sns.despine()
plt.savefig('/work/mech-ai/Mojdeh/iNat_Project-mini-Insecta-2021/Codes/RQ_figures_codes/RQ_modelAcc_vs_OODmethods/RQ_modelAcc_vs_OODmethods.png', bbox_inches = 'tight')
