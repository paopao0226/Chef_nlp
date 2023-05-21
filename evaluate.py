import torch,json
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import random
transformers.logging.set_verbosity_error()

# 加载微调后的模型和tokenizer
path = "/kaggle/input/notebook41f31aa2ad/train_model"
tokenizer = BertTokenizer.from_pretrained(path)
model = BertForSequenceClassification.from_pretrained(path, num_labels=3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载测试数据
data_list = json.load(open('/kaggle/input/chef-test/test_dataset.json', 'r', encoding='utf-8'))
random.shuffle(data_list)

# 解析测试数据
claims = []
documents = []
labels = []
cnt = 0
for item in tqdm(data_list):
    cnt += 1
    claims.append(item["claim"])
    documents.append(item["document"])
    labels.append(item["label"])
    if cnt > 300: break
# 对测试数据进行编码
encoded_inputs = tokenizer(claims, documents, padding=True, truncation=True, return_tensors="pt")

# 获取输入张量和标签
input_ids = encoded_inputs["input_ids"]
attention_mask = encoded_inputs["attention_mask"]
input_labels = torch.tensor(labels)

# 执行评估
with torch.no_grad():
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    predictions = torch.argmax(logits, dim=1)

# 计算准确度和 F1 分数
accuracy = accuracy_score(input_labels, predictions)
f1 = f1_score(input_labels, predictions, average="weighted")

# 输出结果
print(f"准确度：{accuracy:.4f}")
print(f"F1 分数：{f1:.4f}")