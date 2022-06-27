import os
from ipdb import set_trace
import pandas as pd 
from pandas import DataFrame

path = "symptom_de_repeat"
bvas_path = "bvas.xlsx"


filename = [os.path.join(path,file) for file in os.listdir(path)]
H = {}
bvas_df = pd.read_excel("bvas.xlsx")
#所有症状
# for file in filename:
#     f = open(file,"r")
#     data = f.read().splitlines()
#     for i in data:
#         if i in H.keys():
#             H[i] += 1
#         else:
#             H[i] = 1

# data_dict = {"symptom":H.keys(),"value":H.values()}
# df = DataFrame(data_dict)
# df.to_csv("vison/result.csv",index=False)
# set_trace()
for file in filename:
    f = open(file,"r")
    data = f.read().splitlines()
    kkk = []
    for i in data:
        flag_1,flag_2,flag_3,flag_4,flag_5,flag_6,flag_7 = False,False,False,False,False,False,False
        for j in range(len(bvas_df)):
            if ((pd.isnull(bvas_df.body[j]))&(pd.isnull(bvas_df.symptom[j])))|(("高血压"in bvas_df.item[j])|("蛋白尿" in bvas_df.item[j])|("肌酐" in bvas_df.item[j])):
                continue
            if (pd.notnull(bvas_df.body[j]))&(pd.isnull(bvas_df.symptom[j])):
                # if bvas_df.body[j] in i:
                for k in bvas_df.body[j].split(" ")[0:]:
                    if k in i:
                        flag_4=True
                if flag_4==True:
                    flag_7 = True
                    if bvas_df.item[j] not in  H.keys():
                        H[bvas_df.item[j]] = 1
                        kkk.append(bvas_df.item[j])
                    elif bvas_df.item[j] not in kkk:
                        H[bvas_df.item[j]] += 1
                        kkk.append(bvas_df.item[j])
                    break
                else:
                    flag_4=False
                    continue
        
            if (pd.isnull(bvas_df.body[j]))&(pd.notnull(bvas_df.symptom[j])):
                for k in bvas_df.symptom[j].split(" ")[0:]:
                    if k in i:
                        flag_1=True
                if flag_1==True:
                    flag_5=True
                    if  bvas_df.item[j] not in  H.keys():
                        H[bvas_df.item[j]]=1
                        kkk.append(bvas_df.item[j])
                    elif bvas_df.item[j] not in kkk:
                        H[bvas_df.item[j]] += 1
                        kkk.append(bvas_df.item[j])
                    break
                else:
                    flag_1=False
                    continue

            if (pd.notnull(bvas_df.body[j]))&(pd.notnull(bvas_df.symptom[j])):
                # if ("眼睑 浮肿" in i)&("显著突眼" in bvas_df.item[j]):
                #     set_trace()
                for k in bvas_df.symptom[j].split(" ")[0:]:
                    if k in i:
                        flag_2=True
                for m in bvas_df.body[j].split(" ")[0:]:
                    if m in i:
                        flag_3 =True
                if (flag_2==True)&(flag_3==True):
                    flag_6 = True
                    if  bvas_df.item[j] not in  H.keys():
                        H[bvas_df.item[j]]=1
                        kkk.append(bvas_df.item[j])
                        # if bvas_df.item[j]=="缺血性腹痛":
                        #     print(i,file)
                    elif bvas_df.item[j] not in kkk:
                        H[bvas_df.item[j]] += 1
                        kkk.append(bvas_df.item[j])
                        # if bvas_df.item[j]=="缺血性腹痛":
                        #     print(i,file)
                    break
                else:
                    flag_2,flag_3 = False,False
                    continue
        if (flag_7==False)&(flag_5==False)&(flag_6==False):
            if i not in H.keys():
                H[i]=1
            else:
                H[i]+=1
data_dict = {"symptom":H.keys(),"value":H.values()}
df = DataFrame(data_dict)
df.to_csv("vison/result_bvas.csv",index=False)
set_trace()