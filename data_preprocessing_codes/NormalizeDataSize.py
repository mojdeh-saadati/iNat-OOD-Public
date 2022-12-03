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
    print("lastStr ==", lastStr, end = " ")
    strArr.append(lastStr)
    l = len(dir)
    c+=1;
    lenArr.append(l)

    if l > max:
        max = l
print("lenArr ===", lenArr)   
import matplotlib.pyplot as plt  
import pandas as pd;
lenArrdf = pd.DataFrame({'data': lenArr, 'strArr': strArr }) 
print("lenArrdf ====", lenArrdf)
lenArrdf = lenArrdf.sort_values("data");
print("boogh")
lenArrdf.plot.bar(x = 'strArr',y ='data' ,fontsize=7)
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/lenArr.png")
plt.close()

newLen = [min(int(a/max*3000+300), 800) for a in lenArr] 
newLendf = pd.DataFrame({'data': newLen, 'strArr': strArr }) 
newLendf = newLendf.sort_values("data");
newLendf = newLendf.reset_index();
newLendf.at[0:10, 'data'] = 20

print("newLendf ===", newLendf)
newLendf.plot.bar(x = 'strArr',y ='data' ,fontsize=7)

#print(strArr)
plt.savefig("/work/baskarg/Mojdeh/iNat_Project-mini-Insecta-2021/Data/in-distribution/iNat_ISU142_balanced/iNat_agimportant/newLen.png")

avgSize = round(sum(newLen)/142)
print(avgSize)

dict = {};
for count, str in zip(newLendf['data'], newLendf['strArr']):
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
