import os;
import shutil;


path = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/train/";
destinationTrain_b = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/train_b/train/";

for it in os.scandir(path):
    print(it.path);
    try:
        dir = os.listdir(it.path)
    except:
        continue;
    dirPathList=  it.path.split("/")    
    for i,img in enumerate(dir):
        print(i)
        if i >= 411:
            break;
        if i < 411:
            if not os.path.exists(destinationTrain_b+"/"+dirPathList[-1]):
                os.makedirs( destinationTrain_b+"/"+dirPathList[-1], exist_ok=False)
            shutil.copy(it.path +"/"+img, destinationTrain_b+"/"+dirPathList[-1]+"/"+img);

"""
path = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/val/";
destinationInDist = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/inDistOOD/";
destinationTrainOOD = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/trainOOD/";

for it in os.scandir(path):
    print(it.path);
    try:
        dir = os.listdir(it.path)
    except:
        continue;
    dirPathList=  it.path.split("/")    
    for i,img in enumerate(dir):
        print(i)
        if i >= 100:
            if not os.path.exists(destinationInDist+"/"+dirPathList[-1]):
                os.makedirs( destinationInDist+"/"+dirPathList[-1], exist_ok=False)
            shutil.copy(it.path +"/"+img, destinationInDist+"/"+dirPathList[-1]+"/"+img);
        if i < 100:
            if not os.path.exists(destinationTrainOOD+"/"+dirPathList[-1]):
                os.makedirs( destinationTrainOOD+"/"+dirPathList[-1], exist_ok=False)
            shutil.copy(it.path +"/"+img, destinationTrainOOD+"/"+dirPathList[-1]+"/"+img);

"""