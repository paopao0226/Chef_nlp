from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM
) 
import json,re
from zhon import hanzi
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
data = []
import re
def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def preprocessing():
    data_list = json.load(open('train.json', 'r', encoding='utf-8'))
    ev_list = []
    for row in tqdm(data_list):
        claim = row['claim']
        label = row['label']
        for ev in row['evidence'].values():
            sentence = ev['text']
            if(len(sentence) < 5):continue
            d = {}
            d['claim'] = claim
            d['document'] = sentence
            d['label'] = int(label)
            ev_list.append(d)
    jsonArr = json.dumps(ev_list, ensure_ascii=False)
    result = open("dataset.json",'w',encoding='utf-8')
    result.write(jsonArr)
    result.close()
if __name__ == '__main__':
    # preprocessing()
    preprocessing()