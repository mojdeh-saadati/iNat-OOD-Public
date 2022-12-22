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
    
def eval(args):

    val_resize_size, val_crop_size, train_crop_size = 256, 224, 224
    preprocessing = ClassificationPresetEval(
    crop_size=val_crop_size, resize_size=val_resize_size)
    correct = 0;
    total = 0;
    with torch.no_grad():

        ### I need a if clause for to set the model to right format for each model type
        start = time.time()
        if(args.model == "regnet32"):
            model=models.regnet_y_32gf()
            model.fc=torch.nn.Linear(3712,142) 
            torchLoad =torch.load(args.model_path,map_location=torch.device('cpu'))
            model.load_state_dict(torchLoad['model']) 
        if(args.model == 'resnet50'):
            model=torchvision.models.resnet50()
            model.fc=torch.nn.Linear(2048,142)
            torchLoad =torch.load(args.model_path,map_location=torch.device('cpu'))
            model.load_state_dict(torchLoad['model']) 

        if(args.model == 'VGG'):
            model=torchvision.models.vgg11_bn()
            model.classifier._modules['6'] = nn.Linear(4096, 142) 
            torchLoad =torch.load(args.model_path,map_location=torch.device('cpu'))
            model.load_state_dict(torchLoad['model'])


        end = time.time()
        model.eval()
        print("reading the model", end - start)
        start = time.time()
        inDistValid = torchvision.datasets.ImageFolder(args.test_data,preprocessing)
        #test_sampler = torch.utils.data.SequentialSampler(inDistValid)
        inDistValid_loader = DataLoader( inDistValid, batch_size = 256, shuffle = False, num_workers=8, pin_memory=True)
        end = time.time()
        print("reading the initial dataset time", end - start)
        device = torch.device("cuda")
        start = time.time()
        model = model.to(device)
        end = time.time()
        print("transfering model to cuda", end - start)
        
        for inputs, labels in inDistValid_loader:
            """
            start = time.time()
            inputs, labels = inputs.to(device),labels.to(device)
            end = time.time()
            print("importing data", end - start)
            start = time.time()
            logps = model.forward(inputs)
            print(type(logps), "   ", logps.shape)
            predict = torch.argmax(logps,dim=1)
            print("predict ::", type(predict), predict.is_cuda)
            print("labels ::", labels [0], type(labels), labels.is_cuda )
            correct += torch.sum((predict == labels).long())
            total += labels.shape[0] 
            print("accuracy :", float(correct/total))
            end = time.time()
            print("getting the accuracy ", end - start)
            """
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            start = time.time()
            output = model(inputs)
            preds = torch.nn.functional.softmax(output)
            _, preds = torch.max(preds, 1)
            correct += torch.sum((preds == labels).long())
            total += labels.shape[0] 
            end = time.time()
            print("getting the accuracy ", end - start)
            print(labels[0]," accuracy :", float(correct/total))

    acc =  float(correct/total)

    from csv import writer
    print(" final accuracy :", acc)        
    more_lines = [args.model,args.checkpoints,str(acc)]
    with open(args.result_path, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(more_lines)
        f_object.close()


def get_args_parser(add_help=True):

    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--test-data", default="/datasets01/imagenet_full_size/061417/", type=str, help="")    
    parser.add_argument("--result-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--model-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--model", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--checkpoints", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    eval(args) 


