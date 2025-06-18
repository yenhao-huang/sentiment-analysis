import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

class BertWithEmbeddingFusion(nn.Module):
    def __init__(self, model_name="bert-base-uncased", extra_feature_classes=2, num_labels=2):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.extra_embedding = nn.Embedding(extra_feature_classes, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_labels)  # 加上分類頭

    def forward(self, input_ids=None, attention_mask=None, extra_feature=None, labels=None):
        # build custom input embeddings
        token_embeds = self.bert.embeddings.word_embeddings(input_ids)
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device)
        pos_embeds = self.bert.embeddings.position_embeddings(position_ids).unsqueeze(0).expand_as(token_embeds)
        print(extra_feature)
        raise "1"
        extra_embeds = self.extra_embedding(extra_feature).unsqueeze(1).expand_as(token_embeds)

        inputs_embeds = token_embeds + pos_embeds + extra_embeds

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        cls_output = outputs.pooler_output  # [B, H]
        logits = self.classifier(cls_output)  # [B, num_labels]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
