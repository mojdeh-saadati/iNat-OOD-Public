import numpy as np;
import matplotlib.pyplot as plt
import time
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
device = 'cuda'
from torchvision.transforms.functional import InterpolationMode
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



class NewModel(nn.Module):
    def __init__(self, model, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        #print(self.output_layers)
        self.selected_out = OrderedDict()
        #PRETRAINED MODEL
        self.trained_model=model
 
        self.fhooks = []
        for i,l in enumerate(list(self.trained_model._modules.keys())):
            print("i ==",i,"l ===", l)
            if i in self.output_layers:
                self.layer_name = l;
                print("entered the if clause")
                self.fhooks.append(getattr(self.trained_model,l).register_forward_hook(self.forward_hook(l)))
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[self.layer_name] = output
        return hook

    def forward(self, x):
        out = self.trained_model(x)
        return out, self.selected_out



def standalone_get_prelogits(model, dataLoader):
    prelogits_all = []
    logits_all = []
    labels_all = []
    for inputs, labels in dataLoader:
        inputs, labels = inputs.to(device),labels.to(device)
        if len(labels_all) == 0:
            labels_all = labels
        else:
            labels_all =torch.cat((labels_all,labels), dim = 0)   

        model.eval()
        with torch.no_grad():    
            logits, layersOut = model(inputs) 
        #print("layersOut ===",layersOut)
        
        prelogits = layersOut['avgpool']
        if len(logits_all) == 0:
            logits_all  =  logits
        else:
            logits_all  =  torch.cat((logits_all,logits ), dim = 0)   

        if len(prelogits_all) == 0:
            prelogits_all  =  prelogits
        else:
            prelogits_all  =  torch.cat((prelogits_all,prelogits), dim = 0)   
    #return np.concatenate(prelogits_all,axis=0), np.concatenate(logits_all,axis=0), np.concatenate(labels_all,axis=0)
    return prelogits_all.cpu().numpy(), logits_all.cpu().numpy(), labels_all.cpu().numpy()





def extract_prelogit(args):

    print("reading the model")
    model=models.regnet_y_32gf()
    model.fc=torch.nn.Linear(3712,142) 
    torchLoad =torch.load(args.model_path,map_location=torch.device('cuda'))
    model.load_state_dict(torchLoad['model']) 
    print("creating the new class object")
    modelN =NewModel(model , output_layers = [2]).to(device)


    with torch.no_grad():

        # datasets locations
        inDistValidStr = args.indist_test_path
        outDistValidStr = args.outdist_test_path
        inDistTrainStr = args.indist_train_path

        # Loaing the datasets:
        preprocessing = ClassificationPresetEval(crop_size = 224, resize_size = 256)
        inDistValid = torchvision.datasets.ImageFolder(inDistValidStr,preprocessing)
        inDistValid_loader = DataLoader( inDistValid, batch_size = 256, shuffle = False)

        inDistTrain = datasets.ImageFolder(inDistTrainStr,preprocessing)
        inDistTrain_loader = DataLoader(inDistTrain, batch_size = 256, shuffle = False)

        outDistValid = datasets.ImageFolder(outDistValidStr,preprocessing)
        outDistValid_loader = DataLoader(outDistValid, batch_size = 256, shuffle = False)



        print("start prelogit extractions")
        step1 = time.time()
        inDistTrain_embeds, inDistTrain_logits_all, inDistTrain_labels = standalone_get_prelogits(modelN, inDistTrain_loader)
        step2 = time.time()
        print("finish first Duration", step2 - step1)
        torch.save(torch.tensor(inDistTrain_embeds),args.logits_path+args.checkpoints+"inDistTrain_embeds.pt")
        torch.save(torch.tensor(inDistTrain_labels),args.logits_path+args.checkpoints+"inDistTrain_labels.pt")

        step1 = time.time()
        inDistValid_embeds, inDistValid_logits_all, inDistValid_labels = standalone_get_prelogits(modelN, inDistValid_loader)
        step2 = time.time()
        print("finish second Duration", step2 - step1)
        torch.save(torch.tensor(inDistValid_embeds),args.logits_path+args.checkpoints+"inDistValid_embeds.pt")
 
        step1 = time.time()        
        outDistValid_embeds, outDistValid_logits_all, outDistValid_labels = standalone_get_prelogits(modelN, outDistValid_loader)
        step2 = time.time()
        print("finish third Duration", step2 - step1)
        torch.save(torch.tensor(outDistValid_embeds),args.logits_path+args.checkpoints+"outDistValid_embeds.pt")

        

def get_args_parser(add_help=True):

    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--indist-train-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--indist-test-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")    
    parser.add_argument("--outdist-test-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--model-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--checkpoints", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--logits-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    extract_prelogit(args) 