import os
import random
import shutil

path = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/val/"
#destinationMC = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/classifier_train40/"
destinationMV = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/classifier_valid20/"
destinationOT = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_train50/"
destinationOV = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/OOD_valid30/"

#modelClassifierPer = 0.40
modelValidPer = 0.20
OODtrainPer = 0.50
OODvalidPer = 0.30



for it in os.scandir(path):
    sl = it.path.split("/")
    print("ls[-1]", sl[-1])
    #if  sl[-1].find("Insecta") == -1:
    #    continue;
    try:    
        dir = os.listdir(it.path)
    except:
        continue;
    print("len(dir) ==", len(dir))
    #MClen = int(len(dir)*modelClassifierPer)
    MVlen = int(len(dir)*modelValidPer)
    OTlen = int(len(dir)* OODtrainPer) 
    OVlen = int(len(dir)*OODvalidPer)
    theRest = list(range(len(dir)))    
    
    #MCsub = random.sample( theRest,MClen)
    #for x in MCsub:
    #    theRest.remove(x)

    MVsub = random.sample( theRest,MVlen)
    for x in MVsub:
        theRest.remove(x)

    OTsub = random.sample( theRest, OTlen)
    for x in OTsub:
        theRest.remove(x)
    OVsub =  theRest
#    print("len(dir), MCsub, MVsub, OTsub, OVsub  ===", len(dir),"break",MCsub,"break", MVsub,"break" ,OTsub,"break",OVsub)
    print("len(dir), MCsub, MVsub, OTsub, OVsub  ===", len(dir),"break", MVsub,"break" ,OTsub,"break",OVsub)
 
    i = 0;
    for img in dir:
        #if i in MCsub:
        #    if not os.path.exists(destinationMC+"/"+sl[-1]):
        #        os.makedirs( destinationMC+"/"+sl[-1], exist_ok=False)
        #    shutil.copy(it.path +"/"+img, destinationMC+"/"+sl[-1]+"/"+img);
        if i in MVsub:
            if not os.path.exists(destinationMV+"/"+sl[-1]):
                os.makedirs( destinationMV+"/"+sl[-1], exist_ok=False)
            shutil.copy(it.path +"/"+img, destinationMV+"/"+sl[-1]+"/"+img);
        if i in OTsub:
            if not os.path.exists(destinationOT+"/"+sl[-1]):
                os.makedirs( destinationOT+"/"+sl[-1], exist_ok=False)
            shutil.copy(it.path +"/"+img, destinationOT+"/"+sl[-1]+"/"+img);
        if i in OVsub:
            if not os.path.exists(destinationOV+"/"+sl[-1]):
                os.makedirs( destinationOV+"/"+sl[-1], exist_ok=False)
            shutil.copy(it.path +"/"+img, destinationOV+"/"+sl[-1]+"/"+img);     
        i = i+1;    


