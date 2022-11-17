import os
import random
import shutil

insectaPath = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/Others/dataForTest/insecta-2526c-train-copy"
path = "/work/baskarg/iNaturalist/Data/"
destination = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/out-distribution/OODInsect/"
##########################################
###     Extracting Insect Lists
##########################################
labels = []
for it in os.scandir(insectaPath):
    strList = it.path.split('_')
    labels.append((strList[-2]+" "+strList[-1]).strip())

print("labels ===",labels[0], labels[1],len(labels) )

#dir_list = os.listdir(path)
#print(dir_list)
j = 0;
c = 0;
for it in os.scandir(path):
    folderName = it.path.split("/")[-1].strip()
    if folderName in labels :        
        print("boogh"+str(c))
        c = c+1;
        continue  
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
    if len(dir) < 5:
        continue;
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


