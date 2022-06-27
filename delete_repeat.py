import os
from ipdb import set_trace

path = "check"
save_path = "check_de_repeat"

filename = [os.path.join(path,file) for file in os.listdir(path)]

for file in filename:
    flag = file.split("/")[-1]
    fsave = open(os.path.join(save_path,flag),"w+")
    a = []
    f = open(file,"r")
    data = f.read().splitlines()
    data_new = list(set(data))
    for i in data_new:
        fsave.write(i+"\n")
    fsave.close()