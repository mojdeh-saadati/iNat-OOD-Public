from sklearn.metrics import roc_auc_score
from sys import getsizeof
from torch import linalg as LA
import pickle;
import gc
from functools import reduce
import operator as op
import numpy as np;
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
import sys
device = 'cuda'
import pandas as pd



def get_scores(
    indist_train_embeds_in_touse,
    indist_train_labels_in,
    indist_test_embeds_in_touse,
    outdist_test_embeds_in_touse,
    all_train_mean,
    subtract_mean = True,
    normalize_to_unity = True,
    subtract_train_distance = True,
    indist_classes = 142,
    norm_name = "L2",
    ):
#  
##################################################################################################
  # Normalizing the input
  print("indist_train_labels_in inside get_scores ====", indist_train_labels_in)
  description = ""
  
  #indist_train_embeds_in_touse = torch.squeeze(indist_train_embeds_in)
  #indist_test_embeds_in_touse = torch.squeeze(indist_test_embeds_in)
  #outdist_test_embeds_in_touse = torch.squeeze(outdist_test_embeds_in)
  
  df = pd.DataFrame(columns = ["Size","Type", "ObjectSize"] )

  if subtract_mean:
    indist_train_embeds_in_touse = torch.sub(indist_train_embeds_in_touse, all_train_mean)
    indist_test_embeds_in_touse = torch.sub(indist_test_embeds_in_touse, all_train_mean)
    outdist_test_embeds_in_touse = torch.sub(outdist_test_embeds_in_touse, all_train_mean)
    description = description+" subtract mean,"
    print("pass the subtract_mean")



  if normalize_to_unity:
    indist_train_embeds_in_touse = torch.div(indist_train_embeds_in_touse, LA.norm(indist_train_embeds_in_touse,dim=1,keepdims=True))
    indist_test_embeds_in_touse = torch.div(indist_test_embeds_in_touse , LA.norm(indist_test_embeds_in_touse,dim=1,keepdims=True))
    outdist_test_embeds_in_touse = torch.div(outdist_test_embeds_in_touse, LA.norm(outdist_test_embeds_in_touse,dim=1,keepdims=True))
    description = description + " unit norm,"
    print("pass the normalize_to_unity")  


  print("passed first mean")
  
  print("indist_train_embeds_in_touse",indist_train_embeds_in_touse)
  print("indist_test_embeds_in_touse",indist_test_embeds_in_touse)
  print("outdist_test_embeds_in_touse",outdist_test_embeds_in_touse)  
  print("indist_train_labels_in",indist_train_labels_in)
###################################################################################################    
  #Calculate overal mean and covariance of indist train

  maha_intermediate_dict = dict()
  mean = torch.squeeze(torch.mean(indist_train_embeds_in_touse,dim=0))
  meanT = np.cov(torch.t(indist_train_embeds_in_touse - mean).numpy())
  cov = torch.tensor(meanT)
  #cov = cov + 0.01* np.random.rand()
  cov = cov + torch.rand(cov.shape)
  cov_inv = LA.inv(cov) 
  
  print("train in dist mean", meanT)
  print("train in dist cov_inv", cov_inv)

  #getting per class means and average of per class covariance.
  class_means = []
  class_covs_names = []
  class_covs = []
  i = 0;
  sum = []  
  for c in range(indist_classes):
    mean_now = torch.squeeze(torch.mean(indist_train_embeds_in_touse[indist_train_labels_in == c], dim=0))
    print("mean_now ====", mean_now)
    class_means.append(mean_now)
    cov_now = torch.tensor(np.cov(torch.t(indist_train_embeds_in_touse[indist_train_labels_in == c]-mean_now).numpy()))
    if len(sum) == 0:
         sum = cov_now
    else: 
         sum = sum + cov_now
    print("cov_now ===", cov_now)
  sum = sum/indist_classes
  sum = sum + torch.rand(cov.shape)

  class_cov_invs = LA.inv(sum)
###################################################################################################


  maha_intermediate_dict["class_means"] = class_means
  maha_intermediate_dict["mean"] = mean
  maha_intermediate_dict["cov_inv"] = cov_inv
  maha_intermediate_dict["class_cov_invs"] = class_cov_invs

#################################################################################################### 
# We measure two types of distance. 1- distance to whole distribution. 


  out_totrain = maha_distance(outdist_test_embeds_in_touse,cov_inv,mean,norm_name)
  in_totrain = maha_distance(indist_test_embeds_in_touse,cov_inv,mean,norm_name)
  print("out_totrain ====", out_totrain)
  print("in_totrain ====", in_totrain)
  # Since class cov invs is the same for all classes we omit the index. 
  out_totrainclasses = [maha_distance(outdist_test_embeds_in_touse,class_cov_invs,class_means[c],norm_name) for c in range(indist_classes)]
  in_totrainclasses = [maha_distance(indist_test_embeds_in_touse,class_cov_invs,class_means[c],norm_name) for c in range(indist_classes)]
  print("out_totrainclasses ====", out_totrainclasses)
  print("in_totrainclasses ====", in_totrainclasses)

  # 2- distance to nearest class 
  out_scores = torch.min(torch.stack(out_totrainclasses,dim=0),dim=0)[0]
  in_scores = torch.min(torch.stack(in_totrainclasses,dim=0),dim=0)[0]
  print("out_scores ====", out_scores)
  print("in_scores ====", in_scores)
# Normalization
  if subtract_train_distance:
    out_scores = torch.sub(out_scores , out_totrain)
    in_scores = torch.sub(in_scores, in_totrain)

####################################################################################################

  onehots = torch.Tensor([1]*len(out_scores) + [0]*len(in_scores))
  scores = torch.cat((out_scores,in_scores),dim=0)  
  
  return onehots, scores, description, maha_intermediate_dict, in_scores, out_scores


def maha_distance(xs,cov_inv_in,mean_in,norm_type=None):
    diffs = torch.sub(xs, torch.reshape(mean_in,(1,-1)))
    #second_powers = torch.multiply(torch.matmul(diffs.to(torch.float32),cov_inv_in.to(torch.float32)),diffs.to(torch.float32))
    second_powers = torch.matmul(diffs.to(torch.float32),cov_inv_in.to(torch.float32))*diffs.to(torch.float32)

    if norm_type in [None,"L2"]:
        return torch.sum(second_powers,dim=1)
    elif norm_type in ["L1"]:
        return torch.sum(np.sqrt(np.abs(second_powers)),dim=1)
    elif norm_type in ["Linfty"]:
        return torch.max(second_powers,dim=1)

    
    
def MAH(args):
    
    inDistTrain_embeds =  torch.load(args.inDistTrain_embeds)
    all_train_mean = torch.mean(inDistTrain_embeds,dim=0,keepdims=True)
    inDistTrain_embeds = torch.squeeze(inDistTrain_embeds)
    print("One Done")
    inDistTrain_labels = torch.squeeze(torch.load(args.inDistTrain_labels))
    print("Two Done")

    inDistValid_embeds = torch.squeeze(torch.load(args.inDistValid_embeds))
    print("Three Done")

    outDistValid_embeds = torch.squeeze(torch.load(args.outDistValid_embeds))
    print("Four Done")


    df = pd.DataFrame(columns = ["Size","Type", "ObjectSize"] )

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                df = df.append({"name":"{:.13f}".format(id(obj)),"Size":reduce(op.mul,  obj.size() )/(1024*3*8),"Type":type(obj), "ObjectSize" : obj.size()} ,ignore_index = True)
        except:
            pass
    print(df.sort_values(by = ["Size"], ascending=False))    

    print("fifth done")    
    print("inDistTrain_embeds : ",inDistTrain_embeds.element_size() *inDistTrain_embeds.nelement()/(1024**3))
    print("inDistTrain_labels", inDistTrain_labels.element_size() *inDistTrain_labels.nelement()/(1024**3))
    print("inDistValid_embeds", inDistValid_embeds.element_size() *inDistValid_embeds.nelement()/(1024**3))
    print("outDistValid_embeds",outDistValid_embeds.element_size() * outDistValid_embeds.nelement()/(1024**3))

    onehots, scores, description, maha_intermediate_dict, in_scores, out_scores = get_scores(
            inDistTrain_embeds,
            inDistTrain_labels,
            inDistValid_embeds,
            outDistValid_embeds,
            all_train_mean,
            indist_classes = 142,
            subtract_mean = False,
            normalize_to_unity = True,
            subtract_train_distance = True
        )

    onehots = onehots.numpy().reshape((onehots.shape[0], -1))
    scores = scores.numpy().reshape((scores.shape[0], -1))
    torch.save([in_scores, out_scores], args.MAH_path+"/"+args.checkpoint+'MAH.pt')



def get_args_parser(add_help=True):

    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--inDistTrain-embeds", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--inDistTrain-labels", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--inDistValid-embeds", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--outDistValid-embeds", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--MAH-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="")
    parser.add_argument("--checkpoints", default="/datasets01/imagenet_full_size/061417/", type=str, help="")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    MAH(args) 