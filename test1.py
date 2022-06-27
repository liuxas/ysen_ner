import os 
from ipdb import set_trace

ner_dict = {"B_实体_组织获取方式":"B-method", "I_实体_组织获取方式":"I-method", "B_实体_取样位置":"B-sample-body","I_实体_取样位置":"I-sample-body",\
    "B_实体_肿瘤大小":"B-size","I_实体_肿瘤大小":"I-size", "B_实体_组织学类型":"B-type", "I_实体_组织学类型":"I-type","B_实体_组织学级别":"B-class",\
    "I_实体_组织学级别":"I-class","B_实体_MP评级":"B-mp","I_实体_MP评级":"I-mp", "B_实体_送检淋巴结部位":"B-lb-body",\
    "I_实体_送检淋巴结部位":"I-lb-body", "B_实体_送检淋巴结数目":"B-lb-num","I_实体_送检淋巴结数目":"I-lb-num", "B_实体_阳性淋巴结数目":"B-y-num",\
    "I_实体_阳性淋巴结数目":"I-y-num", "B_实体_淋巴结癌变情况":"B-cancer","I_实体_淋巴结癌变情况":"I-cancer","B_实体_淋巴结转移情况":"B-trans", \
    "I_实体_淋巴结转移情况":"I-trans","B_实体_HER-2":"B-her2","I_实体_HER-2":"I-her2","B_实体_ER":"B-er","I_实体_ER":"I-er",\
    "B_实体_PR":"B-pr","I_实体_PR":"I-pr","B_实体_Ki-67":"B-ki67","I_实体_Ki-67":"I-ki67",\
    "B_实体_HER-2扩增":"B-her2-k","I_实体_HER-2扩增":"I-her2-k","B_实体_分子分型":"B-fzfx","I_实体_分子分型":"I-fzfx","B_实体_是否肿瘤":"B-zl","I_实体_是否肿瘤":"I-jz"}

path = "txt1"
save_path = "txt1++"
filename = [os.path.join(path,file) for file in os.listdir(path)]

for file in filename:
    flag = file.split("/")[-1]
    f = open(file,"r")
    data = f.read().splitlines()
    f.close()
    f_save = open(os.path.join(save_path,flag),"w+")
    for i in data:
        tem = i.split("\t")
        # set_trace()
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
    fsave.write("\n")
fsave.close()    

set_trace()