# encoding=utf8
from data_loader import build_dataset
from batch_helper import BatchManager
from data_utils import result_to_json,test_ner
from data_loader import prepare_dataset
import re

import torch
import torch.nn as nn
import torch.optim as optim
from model.LSTM_CRF import NERLSTM_CRF
from sklearn.metrics import f1_score, classification_report

from params import params
from logs.logger import logger
import os,pickle

import time
from datetime import timedelta
from pprint import pprint
from ipdb import set_trace

config = params()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" 记录训练时间 """
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time 
    return timedelta(seconds=int(round(time_dif))) 

def train():
    
    """ 1: 加载数据集，把样本和标签都转化为id"""
    if os.path.isfile(config.data_proc_file):
        with open(config.data_proc_file, "rb") as f:
            train_data,dev_data,test_data = pickle.load(f)
            char_to_id,id_to_char,tag_to_id,id_to_tag = pickle.load(f)
            emb_matrix = pickle.load(f)
        logger.info("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))
            
    else:
        
        train_data,dev_data,test_data, char_to_id, tag_to_id, id_to_tag, emb_matrix = build_dataset()
        
    """ 2: 产生batch训练数据 """
    train_manager = BatchManager(train_data, config.batch_size)
    dev_manager = BatchManager(dev_data, config.batch_size)
    test_manager = BatchManager(test_data, config.batch_size) 
    
    model = NERLSTM_CRF(config, char_to_id, tag_to_id, emb_matrix)
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    """ 3: 用early stop 防止过拟合 """
    total_batch = 0  
    dev_best_f1 = float('-inf')
    last_improve = 0  
    flag = False     
    
    start_time = time.time()
    logger.info(" 开始训练模型 ...... ")
    for epoch in range(config.max_epoch):
        
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.max_epoch))
        
        for index, batch in enumerate(train_manager.iter_batch(shuffle=True)):
            
            optimizer.zero_grad()
            
            """ 计算损失和反向传播 """
            _, char_ids, seg_ids, tag_ids, mask = batch
            loss = model.log_likelihood(char_ids,seg_ids,tag_ids, mask)
            loss.backward()
            
            """ 梯度截断，最大梯度为5 """
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip)
            optimizer.step()
            
            if total_batch % config.steps_check == 0:
                
                model.eval()
                dev_f1,dev_loss = evaluate(model, dev_manager, id_to_tag)
                
                """ 以f1作为early stop的监控指标 """
                if dev_f1 > dev_best_f1:
                    
                    evaluate(model, test_manager, id_to_tag, test=True)
                    dev_best_f1 = dev_f1
                    torch.save(model, os.path.join(config.save_dir,"medical_ner.ckpt"))
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                    
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {} | Dev Loss: {:.4f} | Dev F1-macro: {:.4f} | Time: {} | {}'
                logger.info(msg.format(total_batch, dev_loss, dev_f1, time_dif, improve))  
                
                model.train()
                
            total_batch += 1
            if total_batch - last_improve > config.require_improve:
                """ 验证集f1超过5000batch没上升，结束训练 """
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break                
                

def evaluate_helper(model, data_manager, id_to_tag):

         
    with torch.no_grad():
        
        total_loss = 0
        results = []
        for batch in data_manager.iter_batch():
            
            chars, char_ids, seg_ids, tag_ids, mask = batch
            
            batch_paths = model(char_ids,seg_ids,mask)
            loss = model.log_likelihood(char_ids, seg_ids, tag_ids,mask)
            total_loss += loss.item()    
            
            """ 忽略<pad>标签，计算每个样本的真实长度 """
            lengths = [len([j for j in i if j > 0]) for i in tag_ids.tolist()]
            
            tag_ids = tag_ids.tolist()
            for i in range(len(chars)):
                result = []
                string = chars[i][:lengths[i]]
                
                """ 把id转换为标签 """
                gold = [id_to_tag[int(x)] for x in tag_ids[i][:lengths[i]]]
                pred = [id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]]               
                
                """ 用CoNLL-2000的实体识别评估脚本, 需要按其要求的格式保存结果，
                即 字-真实标签-预测标签 用空格拼接"""
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        
        aver_loss = total_loss / (data_manager.len_data * config.batch_size)        
        return results, aver_loss  
    

def evaluate(model, data, id_to_tag, test=False):
    
    """ 得到预测的标签（非id）和损失 """
    ner_results, aver_loss = evaluate_helper(model, data, id_to_tag)
    
    """ 用CoNLL-2000的实体识别评估脚本来计算F1值 """
    eval_lines = test_ner(ner_results, config.save_dir)
    
    if test:
        
        """ 如果是测试，则打印评估结果 """
        for line in eval_lines:
            logger.info(line)
            
    f1 = float(eval_lines[1].strip().split()[-1]) / 100
    
    return f1, aver_loss


def predict(input_str):
    with open(config.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    
    """ 用cpu预测 """
    model = torch.load(os.path.join(config.save_dir,"medical_ner.ckpt"), 
                       map_location="cpu"
    )
    model.eval()
    
    if not input_str:
        input_str = input("请输入文本: ")    
    
    _, char_ids, seg_ids, _ = prepare_dataset([input_str], char_to_id, tag_to_id, test=True)[0]
    char_tensor = torch.LongTensor(char_ids).view(1,-1)
    seg_tensor = torch.LongTensor(seg_ids).view(1,-1)
    
    with torch.no_grad():
        
        """ 得到维特比解码后的路径，并转换为标签 """
        paths = model(char_tensor,seg_tensor)    
        tags = [id_to_tag[idx] for idx in paths[0]]
    tem = result_to_json(input_str, tags)
    set_trace()
    return tem

def predict_symptom():
    path= "result_txt"
    save_path = "symptom"
    filename = [os.path.join(path,file) for file in os.listdir(path)]
    for file  in filename:
        flag_ = file.split("/")[-1]
        fsave =open(os.path.join(save_path,flag_),"w+")
        f_input = open(file,"r")
        input_str = f_input.read()
        if "鉴别诊断"  in input_str:
            all_min = [r.span() for r in re.finditer('鉴别诊断', input_str)][0][0]
            # all_1 = [r.span() for r in re.finditer('不符', input_str)][-1][1]
            # all_2 = [r.span() for r in re.finditer('不具备', input_str)][-1][1]
            all_3 = [r.span() for r in re.finditer('诊疗计划', input_str)][0][0]
            all_max = all_3
            input_str = input_str[0:all_min]+"\n"+input_str[all_max:]
        all_index = [r.span() for r in re.finditer('\n', input_str)]
        for i in range(len(all_index)):
            if i ==0:
                data_input = input_str[0:all_index[i][1]]
            else:
                data_input = input_str[all_index[i-1][1]:all_index[i][1]]
            flag_1 = True
            # if "今日查房，患者神志清" in data_input:
            #     set_trace()
            result = predict(data_input)
            entity = result["entities"]
            for i in range(len(entity)):
                # if entity[i]["word"] == "呃逆":
                #     set_trace()
                if (entity[i]["type"]=="negative")&(entity[i]["word"]!="主"):
                    flag_1 = False
                    continue
                if i<len(entity)-1:
                    if ((entity[i]["type"]!="symptom")&(entity[i]["type"]!="body")&(entity[i]["type"]!="degree"))|\
                        ((entity[i]["type"]=="body")&((entity[i+1]["type"]=="degree")|(entity[i+1]["type"]=="body"))):
                        flag_1 = True
                else:
                    if (entity[i]["type"]!="symptom")&(entity[i]["type"]!="body")&((entity[i]["type"]!="degree")):
                        flag_1 = True
                if ((entity[i]["type"]=="symptom")|(entity[i]["type"]=="body"))&(("超声"in data_input[entity[i]["start"]-15:entity[i]["start"]])|\
                    (data_input[entity[i]["start"]-1]=="。")|("提示"in data_input[entity[i]["start"]-15:entity[i]["start"]])):
                    flag_1 = True
                if (entity[i]["type"]=="symptom")&(flag_1==True):
                    if i>1:
                        if (entity[i-1]["type"]=="body")&(data_input[entity[i-1]["start"]-1]!="及")&(data_input[entity[i-1]["start"]-1]!="和"):
                            fsave.write(entity[i-1]["word"]+" ")
                        if (entity[i-1]["type"]=="body")&((data_input[entity[i-1]["start"]-1]=="及")|(data_input[entity[i-1]["start"]-1]=="和")\
                            |(data_input[entity[i-1]["start"]-1]=="、"))&(entity[i-2]["type"]=="body"):
                            fsave.write(entity[i-2]["word"]+" ")
                            fsave.write(entity[i-1]["word"]+" ")
                        if ((entity[i-1]["type"] == "degree")|(entity[i-1]["type"] == "disease")|(entity[i-1]["type"] == "may"))&(entity[i-2]["type"]=="body"):
                                fsave.write(entity[i-2]["word"]+" ")
                    if 0<i<2:
                        if entity[i-1]["type"]=="body":
                            fsave.write(entity[i-1]["word"]+" ")
                    fsave.write(entity[i]["word"]+"\n")
        fsave.close()
    return None

def predict_check():
    path= "result_txt"
    save_path = "check"
    filename = [os.path.join(path,file) for file in os.listdir(path)]
    for file  in filename:
        flag_ = file.split("/")[-1]
        fsave =open(os.path.join(save_path,flag_),"w+")
        f_input = open(file,"r")
        input_str = f_input.read()
        if "鉴别诊断"  in input_str:
            all_min = [r.span() for r in re.finditer('鉴别诊断', input_str)][0][0]
            # all_1 = [r.span() for r in re.finditer('不符', input_str)][-1][1]
            # all_2 = [r.span() for r in re.finditer('不具备', input_str)][-1][1]
            all_3 = [r.span() for r in re.finditer('诊疗计划', input_str)][0][0]
            all_max = all_3
            input_str = input_str[0:all_min]+"\n"+input_str[all_max:]
        all_index = [r.span() for r in re.finditer('\n', input_str)]
        for i in range(len(all_index)):
            if i ==0:
                data_input = input_str[0:all_index[i][1]]
            else:
                data_input = input_str[all_index[i-1][1]:all_index[i][1]]
            flag_1 = True
            # if "今日查房，患者神志清" in data_input:
            #     set_trace()
            result = predict(data_input)
            entity = result["entities"]
            for i in range(len(entity)):
                if i==0:
                    if entity[i]["type"]=="result":
                        fsave.write(entity[i]["word"]+"\n")
                else:
                    if 0<i<2:
                        if (entity[i]["type"]=="result")&(entity[i-1]["type"]!="check"):
                            fsave.write(entity[i]["word"]+"\n")
                        if (entity[i]["type"]=="result")&(entity[i-1]["type"]=="check"):
                            fsave.write(entity[i-1]["word"]+" ")
                            fsave.write(entity[i]["word"]+"\n")
                    if i>1:
                        if (entity[i]["type"]=="result")&(entity[i-1]["type"]!="check"):
                            fsave.write(entity[i]["word"]+"\n")
                        if (entity[i]["type"]=="result")&(entity[i-1]["type"]=="check"):
                            if entity[i-2]["type"]!="check":
                                fsave.write(entity[i-1]["word"]+" ")
                                fsave.write(entity[i]["word"]+"\n")
                            else:
                                fsave.write(entity[i-2]["word"])
                                fsave.write(entity[i-1]["word"]+" ")
                                fsave.write(entity[i]["word"]+"\n")           
        fsave.close()
    return None




if __name__ == "__main__":
    
    if config.train:
        train()
    else:
        predict_symptom()
        # predict_check()
