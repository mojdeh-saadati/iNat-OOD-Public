import os
import random
import shutil

path = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/imagenet_2012_copy"
destination = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/imagenet_2012_5000"

#dir_list = os.listdir(path)
#print(dir_list)
j = 0;
for it in os.scandir(path):
    print(j)
    j = j+1;
    #print("it.path ====", it.path);
    try:
        dir = os.listdir(it.path)
        #print("dir ===", dir)
    except:
        continue;
    listChose = range(len(dir))    
    #print("listChose ====", listChose )
    subset = random.sample( listChose, 5)
    #print("len(subset) ====", len(subset))
    #print("subset ====", subset)
    i = 0;
    dirlist=  it.path.split("/")
    for img in dir:
        if i in subset:
            #print("img ===", img)
            #print("mkdir directory", destination+"/"+dirlist[-1])
            #print("it.path ===", it.path)
            if not os.path.exists(destination+"/"+dirlist[-1]):
                os.makedirs( destination+"/"+dirlist[-1], exist_ok=False)
            #print("src ===", it.path +"/"+img)
            #print("dest ===", destination+"/"+dirlist[-1]+"/"+img)
            shutil.copy(it.path +"/"+img, destination+"/"+dirlist[-1]+"/"+img);
        i = i+1;    


