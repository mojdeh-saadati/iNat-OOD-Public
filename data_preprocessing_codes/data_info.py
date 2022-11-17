import numpy as np
from torchvision import datasets, transforms, models
import torch
import argparse
from torch.utils.data import DataLoader
import sys

parser = argparse.ArgumentParser(description='Dataset Info',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--data_add', type=str)
args = parser.parse_args()
inDistValidStr = args.data_add
device='cuda'
data_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])
inDistValid = datasets.ImageFolder( root = inDistValidStr, transform = data_transform)
inDistValid_loader = DataLoader( inDistValid, batch_size = 32, shuffle = False)
inDistValid = inDistValid
print("Dataset Count:", len(inDistValid))
for x,y in inDistValid_loader:
    #print(sys.getsizeof(x[0]),type(x[0]), x[0].element_size() , x[0].nelement(), x[0].shape)
    print("memory size with Gigabyte:", end = "")
    print( x[0].element_size() * x[0].nelement()*len(inDistValid)/(1024*1024*1024))
    print("number of classses:",len(inDistValid.classes))
    break;