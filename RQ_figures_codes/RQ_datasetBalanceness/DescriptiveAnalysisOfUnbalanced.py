import sys;
import os;
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt

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
from torchvision.transforms.functional import InterpolationMode
import time
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class ClassificationPresetEval:
    def __init__(
        self,
        crop_size = 224,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)


destinationTrain_ub_uniform = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/train_ub_uniform/train/";
destinationTrain_ub = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/train_ub/train/";

modelPath_ub_uniform  = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub_uniform/checkpoints/model_49.pth"
modelPath_ub  = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Models/Regnet32_balancedSubset_ub/checkpoints/model_49.pth"

inDistPath = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/inDistOOD/"
outDistPath = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/inDistOOD/"

path = destinationTrain_ub_uniform;
model_path = modelPath_ub_uniform;

des_table = pd.DataFrame();
for it in os.scandir(path):
    fN = it.path.split("/")
    fN = fN[-1]
    dir = os.listdir(it.path)
    print(fN, len(dir), end  = "")
    temp = pd.DataFrame({'folderName':[fN], 'folderSize':[len(dir)], 'classAcc':[0], 'classFPOutDist':[0]})
    des_table= pd.concat([des_table, temp], ignore_index=True) 
print(des_table)    


##############################################################################
#### calculating accuracy per class
##############################################################################
val_resize_size, val_crop_size, train_crop_size = 256, 224, 224
preprocessing = ClassificationPresetEval(
crop_size = val_crop_size, resize_size = val_resize_size)
with torch.no_grad():
    model=models.regnet_y_32gf()
    model.fc=torch.nn.Linear(3712,142) 
    torchLoad =torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(torchLoad['model']) 
    model.eval()
    inDistValid = torchvision.datasets.ImageFolder(inDistPath,preprocessing)
    #print(inDistValid.class_to_idx)

    inDistValid_loader = DataLoader( inDistValid, batch_size = 256, shuffle = False, num_workers=8, pin_memory=True)
    device = torch.device("cuda")
    model = model.to(device)
    labelSizePerClass = {};
    correctPerdPerClass = {};
    accPerClass = [];
    for inputs, labels in inDistValid_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(inputs)
        preds = torch.nn.functional.softmax(output)
        _, preds = torch.max(preds, 1)
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()
        for l in labels:
            if l in labelSizePerClass.keys():
                labelSizePerClass[l] +=1;
            else:
                labelSizePerClass[l] = 1;
        for i in range(len(labels)):
            if preds[i] == labels[i]:
                if labels[i] in correctPerdPerClass.keys():
                    correctPerdPerClass[labels[i]] +=1;
                else:
                    correctPerdPerClass[labels[i]] =1 
    """
    for key in labelSizePerClass:
        if key in correctPerdPerClass:
            accPerClass[key] = float(correctPerdPerClass[key]/labelSizePerClass[key]) 
        else:
            accPerClass[key] = 0;     

    """    
    for i in range(142):
        if i in correctPerdPerClass:
            accPerClass.append(float(correctPerdPerClass[i]/labelSizePerClass[i])) 
        else:
            accPerClass.append(0);     

    #print(accPerClass)
print("dataframe .shape ===", des_table.shape)
des_table = des_table.sort_values(by = ["folderName"])


des_table["classAcc"]  = accPerClass
print(des_table)
##############################################################################
#### calculating false positive per class
##############################################################################

with torch.no_grad():
    model=models.regnet_y_32gf()
    model.fc=torch.nn.Linear(3712,142) 
    torchLoad =torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(torchLoad['model']) 
    model.eval()
    outDistValid = torchvision.datasets.ImageFolder(outDistPath,preprocessing)
    outDistValid_loader = DataLoader( outDistValid, batch_size = 256, shuffle = False, num_workers=8, pin_memory=True)
    device = torch.device("cuda")
    model = model.to(device)
    totalDataSize = 0;
    FPperClass = [0]*142;
    for inputs, _ in outDistValid_loader:
        inputs = inputs.to(device, non_blocking=True)
        output = model(inputs)
        preds = torch.nn.functional.softmax(output)
        _, preds = torch.max(preds, 1)
        preds = preds.cpu().numpy()
        totalDataSize += len(preds)
        for p in preds:
            if p in FPperClass:            
                FPperClass[p] +=1;
            else:
                FPperClass[p] =1        
for i in range(142):
    FPperClass[i] = FPperClass[i]/totalDataSize;

des_table["classFPOutDist"]  = FPperClass

des_table = des_table.sort_values(by = ["classFPOutDist"])
print(des_table)
