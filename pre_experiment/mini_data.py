import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

# 标签定义 (BC5CDR 只包含 Chemical 和 Disease)
LABELS = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

class BC5CDRDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        # 尝试使用不同路径兼容当前工作目录的不同情况
        if not os.path.exists(data_path) and os.path.exists(data_path.replace("../", "")):
            data_path = data_path.replace("../", "")
            
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        entities = item['entities']

        # 1. 构建字符级别的标签，用于后续对齐
        char_labels = ["O"] * len(text)
        for ent in entities:
            mention, start, end, ent_type, _ = ent
            if start < len(text) and end <= len(text):
                char_labels[start] = f"B-{ent_type}"
                for i in range(start + 1, end):
                    char_labels[i] = f"I-{ent_type}"

        # 2. Tokenize 文本，并返回 offset_mapping 来对齐 token 和 字符
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        # 3. 对齐标签 (Token 级别)
        labels = []
        offsets = encoding.pop("offset_mapping")[0].numpy()
        for start, end in offsets:
            if start == 0 and end == 0:
                # [CLS]、[SEP]、[PAD] 等特殊字符
                labels.append(-100) 
            else:
                # 取 token 第一个字符的标签作为该 token 的标签
                token_label = char_labels[start]
                labels.append(LABEL2ID[token_label])
                
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoding

def get_dataloader(data_path, tokenizer, batch_size=4, shuffle=True, max_len=128):
    if not os.path.exists(data_path):
        # 尝试一些默认路径修复
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "data_pre", "data_pred", "BC5CDR", os.path.basename(data_path))
        if os.path.exists(alt_path):
            data_path = alt_path
    dataset = BC5CDRDataset(data_path, tokenizer, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
