import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformers import AutoTokenizer
from models.bert_with_extra_features import BertWithEmbeddingFusion
import torch

def test_model():
    # 1. 初始化 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertWithEmbeddingFusion(model_name="bert-base-uncased", extra_feature_classes=2, num_labels=2)

    # 2. 假資料（batch_size=2）
    texts = ["I love cats", "This is amazing"]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    input_ids = inputs["input_ids"]                    # [2, seq_len]
    attention_mask = inputs["attention_mask"]          # [2, seq_len]
    extra_feature = torch.tensor([0, 1])               # 假設兩筆對應 extra class
    labels = torch.tensor([0, 1])                      # 假設真實類別

    # 3. forward 測試
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, extra_feature=extra_feature, labels=labels)

    # 4. 印出結果
    print("Logits:", outputs.logits)         # 應該是 [2, 2]
    print("Loss:", outputs.loss)             # 應該是一個 scalar tensor

test_model()
