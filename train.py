import torch,json
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
transformers.logging.set_verbosity_error()

model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        claim = self.data[index]['claim']
        document = self.data[index]['document']
        label = self.data[index]['label']

        encoding = self.tokenizer.encode_plus(
            claim,
            document,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }
def train(model, train_dataloader, num_epochs, learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            print("epoch:",epoch,",loss:",loss)
        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {average_loss}")

    print("Training finished!")

    return model

def predict(claim, document):
    inputs = tokenizer.encode_plus(claim, document, add_special_tokens=True, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 将输入移动到GPU（如果可用）
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    trained_model.to(device)
    # 将模型设置为评估模式
    trained_model.eval()

    # 使用输入进行前向传播
    with torch.no_grad():
        outputs = trained_model(input_ids, attention_mask=attention_mask)

    # 获取预测的标签
    logits = outputs.logits
    _, predicted_labels = torch.max(logits, dim=1)

    return predicted_labels.item()


# 加载训练数据集
# train_data = [
#     {'claim': '截至2020年3月底，猪肉价格持续小幅下跌。', 'document': '事实文本...', 'label': 0},
#     {'claim': '另一种声明...', 'document': '另一段事实文本...', 'label': 1},
#     # 更多数据样本...
# ]
train_data = json.load(open('dataset.json', 'r', encoding='utf-8'))
dataset = CustomDataset(train_data)
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
model.to(device)
# 开始微调训练
num_epochs = 3
learning_rate = 2e-5
trained_model = train(model, train_dataloader, num_epochs, learning_rate)

fact_text = "中新网客户端北京11月19日电(记者李金磊)人民币对美元汇率升到6.5时代，早前换了美元的人已经哭晕在厕所。人民币对美元汇率升到6.5时代时隔两年多，人民币对美元汇率重回6.5元时代。数据显示，11月17日人民币对美元汇率中间价为6.5762，较上一交易日上升286基点，升到了6.5元时代。此外，在岸、离岸人民币对美元汇率均涨到了6.5元时代。此后，人民币对美元汇率在6.5元区间继续大幅升值。11月19日，人民币对美元汇率中间价为6.5484，较上一交易日上涨109基点，升破了6.55关口。早前换美元的已哭晕在厕所整体来看，今年人民币汇率气势如虹，从5月底开始一路走强。5月29日人民币对美元汇率中间价为7.1316，11月19日则为6.5484，短短近6个月时间，人民币对美元汇率中间价升值了5832个基点。从7.1316到6.5484，如果换汇1万美元，当时需要7.1316万人民币，今天则只需6.5484万人民币，可以省下5832元人民币。所以，如果你是之前换了美元的话，可以说是亏了不少。多重因素刺激人民币汇率走强“近期人民币升值，主要是受到经济基本面的支撑。”国家外汇管理局副局长、新闻发言人王春英10月23日回应称，中国率先控制了疫情，经济社会恢复发展，国际货币基金组织预测今年中国将是唯一实现经济正增长的主要经济体。出口形势良好，境外长期资金有序增持人民币资产，人民币汇率在市场供求推动下有所升值。中国外汇投资研究院院长、首席经济学家谭雅玲对中新网记者表示，近期多重因素又推动人民币汇率新一波上涨，包括第三届进博会召开、区域全面经济伙伴关系协定(RCEP)正式签署。RCEP是目前全球体量最大的自贸区，2019年，RCEP的15个成员国总人口达22.7亿，GDP达26万亿美元，出口总额达5.2万亿美元，均占全球总量约30%。“人民币对美元汇率走进6.5时代，也是汇率市场投机情绪的反映。”谭雅玲称，中国金融市场加速开放，所以国际资本在投机、对冲和套利方面比较青睐中国市场，这种叠加效应就促使了资金涌入中国市场。不过，谭雅玲提醒，美元价格是自由浮动，而人民币是有限浮动，但人民币的升值幅度却超过了美元的贬值幅度。这种异常现象背后，可能有海外对冲基金刻意打压中国利润的嫌疑。汇兑损失令外贸企业承压中国民生银行首席研究员温彬对中新网记者表示，人民币升值有利于进口，进口企业会降低采购成"
claim = "2020年11月30日人民币对美元汇率中间价下调27个基点。"

predicted_label = predict(claim, fact_text)
print("Predicted Label:", predicted_label)
trained_model.save_pretrained("train_model")
tokenizer.save_pretrained("train_model")
