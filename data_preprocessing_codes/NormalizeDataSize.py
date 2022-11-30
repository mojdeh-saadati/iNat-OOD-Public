import os;

path = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142/classifier_train/";
max = -1;
lenArr = [];
strArr = [];
c  = 1;
for it in os.scandir(path):
    dir = os.listdir(it.path)
    strL = str(it.path).split("/")
    lastStr = strL[-1];
    strArr.append(lastStr)
    l = len(dir)
    c+=1;
    lenArr.append(l)

    if l > max:
        max = l
print("lenArr ===", lenArr)        
newLen = [int(a/max*200+600) for a in lenArr] 
print(strArr)
print(newLen)
avgSize = round(sum(newLen)/142)
print(avgSize)

dict = {};
for count, str in zip(newLen, strArr):
    dict[str] = count;
print(dict)




import shutil;
path = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/train/";
destination = "/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/train_ub/";

for it in os.scandir(path):
    print(it.path);
    try:
        dir = os.listdir(it.path)
    except:
        continue;
    dirPathList=  it.path.split("/")    
    folderName = dirPathList[-1];
    print("folderName ===", folderName)
    print("count === ", dict[folderName])
    for i,img in enumerate(dir):

        if i >= dict[folderName]:
            break;
        if not os.path.exists(destination+"/"+dirPathList[-1]):
                os.makedirs( destination+"/"+dirPathList[-1], exist_ok=False)
        
        shutil.copy(it.path +"/"+img, destination+"/"+dirPathList[-1]+"/"+img);


