import os
import re
from collections import Counter

import pandas as pd
from ipdb import set_trace
import numpy  as  np

#将所有词汇总在all_txt
# paths = ["check","symptom"]

# def function1(paths):
#     file_lists=[]
#     for path in paths:
#         filename = [os.path.join(path,file) for file in os.listdir(path)]
#         file_lists.append(filename)
#     return file_lists

# def function2(filelists):
#     for filename in filelists:
#         flag = filename[0].split("/")[0]
#         fwrite = open("all_txt/"+flag+".txt","w+")
#         for file in filename:
#             fdata = open(file)
#             data = fdata.read()
#             fwrite.write(data)
#     return None

# file_list = function1(paths)
# function2(file_list)
# set_trace()

path ="all_txt"
files = [os.path.join(path,file) for file in os.listdir(path)]
 
def function1(files):
    result_list=[]
    for file in files:
        fdata = open(file)
        data = fdata.read().splitlines()
        result=Counter(data) 
        result_list.append(result)
    return result_list

def function2(result):
    H = {}
    data_bvas = pd.read_excel("bvas.xlsx")
    data = result.keys()
    for i in data:
        for j in range(len(data_bvas)):
            if pd.isnull(data_bvas.iloc[j].body)&pd.isnull(data_bvas.iloc[j].symptom):
                continue
            if pd.notnull(data_bvas.iloc[j].body)& pd.isnull(data_bvas.iloc[j].symptom):
                flag = False
                for k in data_bvas.iloc[j].body.split(" "):
                    if k in i:
                        flag = True
                if flag == True:
                    if data_bvas.iloc[j].itemss in H.keys():
                        H[data_bvas.iloc[j].itemss] +=  result[i]
                    else:
                        H[data_bvas.iloc[j].itemss] = result[i]
                    break
                else:
                    continue
            if pd.isnull(data_bvas.iloc[j].body)&pd.notnull(data_bvas.iloc[j].symptom):
                flag1 = False
                for k in data_bvas.iloc[j].symptom.split(" "):
                    if k in i:
                        flag1 = True
                if flag1==True:
                    try:
                        if data_bvas.iloc[j].itemss in H.keys():
                            H[data_bvas.iloc[j].itemss] += result[i]
                        else:
                            H[data_bvas.iloc[j].itemss] = result[i]
                    except:
                        set_trace()
                    break
                else:
                    continue
            if pd.notnull(data_bvas.iloc[j].body)&pd.notnull(data_bvas.iloc[j].symptom):
                flag2 = False
                flag3 = False
                flag4 = False
                for k in data_bvas.iloc[j].symptom.split(" "):
                    if k in i:
                        flag2 = True
                for k in data_bvas.iloc[j].body.split(" "):
                    if k in i:
                        flag3 = True
                if flag2==True&flag3==True:
                    flag4 = True
                if flag4 == True:
                    # set_trace()
                    if data_bvas.iloc[j].itemss in H.keys():
                        H[data_bvas.iloc[j].itemss]  += result[i]
                    else:
                        H[data_bvas.iloc[j].itemss] = result[i]
                    break
                else:
                    continue
        if flag==False&flag1==False&flag4==False:
            if i in H.keys():
                H[i] += result[i]
            else:
                H[i] = result[i]
    return H

def function3(result):
    H = {}
    data_bvas = pd.read_excel("bvas.xlsx")
    data = result.keys()
    for i in data:
        flag = False
        if "肌酐" in i:
            pattern = re.compile(r'\d+')   # 查找数字
            try:
                tem = pattern.findall(i)[0]
            except:
                 print("this check have not number.")
            value = int(tem)
            if 125<=value<=249:
                if "肌酐125-249umol"  not in H.keys():
                    H["肌酐125-249umol"] = 1 
                else:
                    H["肌酐125-249umol"] += 1
            elif 250<=value<=499:
                if "肌酐250-499umol"  not in H.keys():
                    H["肌酐250-499umol"] = 1 
                else:
                    H["肌酐250-499umol"] += 1
            elif 500<=value:
                if "肌酐>500umol"  not in H.keys():
                    H["肌酐>500umol"] = 1 
                else:
                    H["肌酐>500umol"] += 1
            else:
                continue
        elif ("蛋白" in i)&("尿" in i):
            if ("阳性" in i)|("++" in  i)|("+++" in i)|("++++" in i)|("2+" in i)|("3+" in i)|("4+" in i):
                if "蛋白尿定性" not in H.keys():
                    H["蛋白尿定性"] = 1
                else:
                    H["蛋白尿定性"] +=1
            else:
                continue
        else:
            continue
        # if flag==False&flag1==False&flag4==False:
        #     if i in H.keys():
        #         H[i] += result[i]
        #     else:
        #         H[i] = result[i]
    return H


result_list = function1(files)
# H = function2(result_list[1])
# df_dict = {"symptom":H.keys(), "num":H.values()}
# df = pd.DataFrame(df_dict)
# df.to_csv("bing_bvas.csv",index=False)

# H = function3(result_list[0])
# df_dict = {"check":H.keys(), "num":H.values()}
# df = pd.DataFrame(df_dict)
# df.to_csv("check_bvas.csv",index=False)

