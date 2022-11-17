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
#from OOD_methods import SMT
# This file can be archived


def run_model(args):
    # datasets locations
    inDistValidStr = args.indist_test_path
    outDistValidStr = args.indist_test_path
    inDistTrainStr = args.indist_train_path


    # Loaing the datasets:
    data_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])

    inDistValid = datasets.ImageFolder( root = inDistValidStr, transform = data_transform)
    inDistValid_loader = DataLoader( inDistValid, batch_size = 256, shuffle = False)

    inDistTrain = datasets.ImageFolder( root = inDistTrainStr, transform = data_transform)
    inDistTrain_loader = DataLoader(inDistTrain, batch_size = 256, shuffle = False)

    outDistValid = datasets.ImageFolder(root = outDistValidStr, transform = data_transform)
    outDistValid_loader = DataLoader(inDistTrain, batch_size = 256, shuffle = False)


    for name in modelsNames:
    #    print(name);
        modelName = modelsPath + name 
        # Load the corresponding 
        print(modelName)
        trained_model = models.regnet_y_32gf()
        trained_model.fc = torch.nn.Linear(3712,142) 
        torchLoad = torch.load( modelName,map_location=torch.device('cuda'))
        trained_model.load_state_dict(torchLoad['model']) 
        model = trained_model
        model0 = returnModel(modelName, "SMT")
        model1 = returnModel(modelName, "MAH")

        #AUROC_S, AUPR_S,FP95_S ,AUROC_E, AUPR_E ,FP95_E = SMT_EBM(model0, inDistTrain_loader, inDistValid_loader, outDistValid_loader)
        #print(AUROC_S, AUPR_S,FP95_S ,AUROC_E, AUPR_E ,FP95_E)
        #print("inDistTrain_labels befpre mah ====", inDistTrain_labels)
        AUROC_M, AUPR_M ,FP95_M = MAH(model1, inDistTrain_embeds,inDistTrain_labels,inDistValid_embeds,outDistValid_embeds)
        #figs[i][j,:] = AUROC1, AUROC2, AUROC3



def get_args_parser(add_help=True):

    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--indist-train-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--indist-test-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")    
    parser.add_argument("--outdist-test-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--logits-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run_model(args)