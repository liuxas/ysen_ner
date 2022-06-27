import os 
from ipdb import set_trace

ner_dict = {"B_实体_部位":"B-body", "I_实体_部位":"I-body", "B_实体_症状":"B-symptom","I_实体_症状":"I-symptom",\
    "B_实体_检查":"B-check","I_实体_检查":"I-check", "B_实体_检查结果":"B-result", "I_实体_检查结果":"I-result","B_实体_疾病":"B-disease",\
    "I_实体_疾病":"I-disease","B_实体_否定词":"B-negative","I_实体_否定词":"I-negative", "B_实体_可能性词":"B-may",\
    "I_实体_可能性词":"I-may", "B_实体_程度词":"B-degree","I_实体_程度词":"I-degree", "B_实体_条件词":"B-condition",\
    "I_实体_条件词":"I-condition", "B_实体_频率词":"B-frequency","I_实体_频率词":"I-frequency","B_实体_既往词":"B-history", \
    "I_实体_既往词":"I-history","B_实体_时间词":"B-time","I_实体_时间词":"I-time","B_实体_药物":"B-drug","I_实体_药物":"I-drug",\
    "B_实体_条件词":"B-condition","I_实体_条件词":"I-condition","B_实体_治疗":"B-treatment","I_实体_治疗":"I-treatment",\
    "B_实体_手术":"B-operation","I_实体_手术":"I-operation"}

path = "txt"
save_path = "txt++"
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