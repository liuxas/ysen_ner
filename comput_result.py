import os 
from ipdb import set_trace
import pandas as pd 
import re

path= "check_de_repeat"
bvas_path = "bvas.xlsx"

filename = [os.path.join(path,file) for file in os.listdir(path)]
H = {}
bvas_df = pd.read_excel("bvas.xlsx")

num=0
for file in filename:
    num+=1
    value = 0
    f = open(file,"r")
    data = f.read().splitlines()
    flag = False
    for i in data:
        if "肌酐" in i:
            pattern = re.compile(r'\d+')   # 查找数字
            try:
                tem = pattern.findall(i)[0]
            except:
                print("this check have not number.")
            if int(tem)>value:
                value = int(tem)
        if ("蛋白" in i)&("尿" in i):
            print(i,file)
            if ("阳性" in i)|("++" in  i)|("+++" in i)|("++++" in i)|("2+" in i )|("3+" in i)|("4+" in i):
                flag = True
    if 125<=value<=249:
        if "肌酐125-249umol"  not in H.keys():
            H["肌酐125-249umol"] = 1 
        else:
            H["肌酐125-249umol"] += 1
    if 250<=value<=499:
        if "肌酐250-499umol"  not in H.keys():
            H["肌酐250-499umol"] = 1 
        else:
            H["肌酐250-499umol"] += 1
    if 500<=value:
        if "肌酐>500umol"  not in H.keys():
            H["肌酐>500umol"] = 1 
        else:
            H["肌酐>500umol"] += 1

    if flag == True:
        if "蛋白尿定性" not in H.keys():
            H["蛋白尿定性"] = 1
        else:
            H["蛋白尿定性"] +=1


data_dict = {"symptom":H.keys(),"value":H.values()}
df = pd.DataFrame(data_dict)
df.to_csv("vison/result_shen.csv",index=False)

set_trace()