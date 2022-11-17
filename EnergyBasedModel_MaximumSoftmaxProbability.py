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

device = 'cuda'
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



def MSP_EBM(args):

    print("reading the model")
    ### I need a if clause for to set the model to right format for each model type
    if(args.model == "regnet32"):
        model=models.regnet_y_32gf()
        model.fc=torch.nn.Linear(3712,142) 
        torchLoad =torch.load(args.model_path,map_location=torch.device('cuda'))
        model.load_state_dict(torchLoad['model']) 
    if(args.model == 'resnet50'):
        model=torchvision.models.resnet50()
        model.fc=torch.nn.Linear(2048,142)
        torchLoad =torch.load(args.model_path,map_location=torch.device('cuda'))
        model.load_state_dict(torchLoad['model']) 
            #weights=torch.load('imagenet-21k-weights',map_location=torch.device('cuda'))['model']
            #Format: torch.nn.Linear(Dimension of latent space, Number of classes)

    # datasets locations
    inDistValidStr = args.indist_test_path
    outDistValidStr = args.outdist_test_path

    # Loaing the datasets:
    preprocessing = ClassificationPresetEval(crop_size = 224, resize_size = 256)
    
    
    inDistValid = datasets.ImageFolder( inDistValidStr, preprocessing)
    inDistValid_loader = DataLoader( inDistValid, batch_size = 256, shuffle = False, num_workers=8, pin_memory=True)
    outDistValid = datasets.ImageFolder(outDistValidStr, preprocessing)
    outDistValid_loader = DataLoader(outDistValid, batch_size = 256, shuffle = False, num_workers=8, pin_memory=True)

        
    # No inference is needed. Prediction  
    # In Distribution data values    
    
    print("DataSizes: inDistValid", len(inDistValid))
    print("DataSizes: outDistValid", len(outDistValid))
    
    ################################################################################    
    device = torch.device("cuda")
    model.to(device)
    softmax_indist = []
    energy_indist = []
    inDistValid_y = []
    test_loss = 0
    accuracy = 0
    model.eval()
    model = model.to(device)
    CT = 0;
    T = 1;
    with torch.no_grad():
        for inputs, labels in inDistValid_loader:
            inputs, labels = inputs.to(device, non_blocking=True),labels.to(device,non_blocking=True)
            if len(inDistValid_y) == 0:
                inDistValid_y = labels
            else:
                inDistValid_y =torch.cat((inDistValid_y,labels), dim = 0)   
            
            logps = model(inputs)
            sf= torch.nn.Softmax(dim=1)
            softmax = sf(logps)
            softmax = softmax
            if len(softmax_indist) == 0:
                softmax_indist  =  softmax
            else:
                softmax_indist  =  torch.cat((softmax_indist,softmax), dim  = 0)  
            energy = -(T*torch.logsumexp(logps / T, dim=1))   
            energy =torch.reshape(energy, (-1, 1))
            if len(energy_indist) == 0:
                energy_indist  =  energy
            else:
                energy_indist  = torch.cat((energy_indist,energy), dim = 0)          
    print("finished-IN")    
    # Out distribution data values     
    ################################################################################
    
    device = torch.device("cuda")
    model.to(device)
    softmax_outdist = []
    energy_outdist = []
    outDistValid_y = []
    test_loss = 0
    accuracy = 0
    model.eval()
    model = model.to(device)
    CT = 0;
    with torch.no_grad():
        for inputs, labels in outDistValid_loader:
#            CT = CT + 1;
#            if CT > 10:
#                break;
            inputs, labels = inputs.to(device, non_blocking = True),labels.to(device, non_blocking = True)
            if len(outDistValid_y) == 0:
                outDistValid_y = labels
            else:
                outDistValid_y = torch.cat((outDistValid_y,labels), dim = 0)   
            
            logps = model(inputs)
            sf= torch.nn.Softmax(dim=1)
            softmax = sf(logps)
            if len(softmax_outdist) == 0:
                softmax_outdist  =  softmax
            else:
                softmax_outdist  =  torch.cat((softmax_outdist,softmax), dim = 0)        
                
            energy = -(T*torch.logsumexp(logps / T, dim=1))  
            energy = torch.reshape(energy, (-1, 1))
            if len(energy_outdist) == 0:
                energy_outdist  =  energy
            else:
                energy_outdist  =  torch.cat((energy_outdist,energy), dim = 0)  


   
    print("start entropy stat")
    s_prob_in  = torch.amax(softmax_indist, dim=1, keepdims=True)
    s_prob_out = torch.amax(softmax_outdist, dim=1, keepdims=True)


    print("len(softmax indist)", softmax_indist.shape)
    print("len(softmax outdist)", softmax_outdist.shape)

    print("len(softmax indist)", energy_indist.shape)
    print("len(softmax outdist)", energy_outdist.shape)

    print("len(s_prob_in)", s_prob_out.shape)
    print("len(s_prob_out)", s_prob_out.shape)

    torch.save([s_prob_in, s_prob_out], args.MSP_path+"/"+args.checkpoint+'MSP.pt')
    torch.save([energy_indist, energy_outdist], args.EBM_path+"/"+args.checkpoint+'EBM.pt')
 




def get_args_parser(add_help=True):

    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--indist-test-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")    
    parser.add_argument("--outdist-test-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--model-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--checkpoint", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--MSP-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--EBM-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--model", default="/datasets01/imagenet_full_size/061417/", type=str, help="")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    MSP_EBM(args) 