import os 
from ipdb import set_trace

ner_dict = {"B_实体_HER-2扩增":"B_实体_HER-2","I_实体_HER-2扩增":"I_实体_HER-2"}

path = "ann"
save_path = "ann2"
filename = [os.path.join(path,file) for file in os.listdir(path)]

for file in filename:
    flag = file.split("/")[-1]
    f = open(file,"r")
    data = f.read().splitlines()
    f.close()
    f_save = open(os.path.join(save_path,flag),"w+")
    for i in data:
        tem = i.split("\t")
        if tem[1] in ner_dict.keys():
            new_line = tem[0]+" "+ner_dict[tem[1]]+"\n"
            f_save.write(new_line)
        elif tem[0]=="_换行符_":
            new_line = "\n"
            f_save.write(new_line)
        else:
            new_line = tem[0]+" "+tem[1]+"\n"
            f_save.write(new_line)
    f_save.close()

filename = [os.path.join(save_path,file) for file in os.listdir(save_path)]

fsave = open(os.path.join(save_path,"example_new.train"),"w+")
for file in filename:
    f = open(file,"r")
    data = f.read()
    fsave.write(data)
fsave.close()    

set_trace()