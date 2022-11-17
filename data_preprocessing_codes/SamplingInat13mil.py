import os
import random
import shutil

path = "/work/baskarg/iNaturalist/Data/"
destination = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution-valid/insectOOD/"

#dir_list = os.listdir(path)
#print(dir_list)

for it in os.scandir(path):
    print(it.path);
    try:
        dir = os.listdir(it.path)
    except:
        continue;
# Checking if the list is empty or not
    if len(dir) != 0:
        print("inside else")
        r = random.uniform(0, 1)
        source = it.path;
        if r <= 0.01:
            print("source", source)
            sl = source.split("/")
            shutil.copytree(source, destination+"/"+sl[-1])
            