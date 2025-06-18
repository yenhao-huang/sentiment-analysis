## 🎯 Goal

Fine-tune a sentiment analysis model on multiple datasets and evaluate its generalization performance.

---

## 📦 Datasets

| Dataset                     | Size  | Task Type             | Description                                |
| --------------------------- | ----- | --------------------- | ------------------------------------------ |
| IMDb Large Movie Review     | 50K   | Binary Sentiment      | Movie reviews labeled as positive/negative |
| dair-ai/emotion             | \~16K | Multi-class Emotion   | Texts labeled with 6 basic emotions        |
| Sp1786/multiclass-sentiment | \~8K  | Multi-class Sentiment | Sentiment ratings: strongly neg → pos      |

---

Sure! Here's the English version of your pretrained model section, with structure and training data details added:

---

## 🔧 Pretrained Models

### `bert-base-uncased`

It is based on Google’s original **BERT** architecture introduced in 2018.

* **Model Architecture**:

  * Layers: 12 Transformer encoder layers
  * Hidden size: 768
  * Attention heads: 12
  * Total parameters: \~110 million

* **Training Corpus**:

  * **BooksCorpus** (800 million words)
  * **English Wikipedia** (2.5 billion words)
  * Total: \~3.3 billion words covering diverse topics

---

## ⚙️ Training Configuration (Trainer Arguments)

### Optimizer & Learning Rate

| Argument            | Description                                             |
| ------------------- | ------------------------------------------------------- |
| `learning_rate`     | Initial learning rate (e.g., `2e-5`)                    |
| `weight_decay`      | L2 regularization (used with `AdamW`)                   |
| `lr_scheduler_type` | LR schedule strategy (`linear`, `cosine`, `polynomial`) |
| `warmup_steps`      | Linear warmup steps (helps stabilize early training)    |
| `adam_beta1/2`      | Momentum terms in Adam optimizer                        |
| `adam_epsilon`      | Small constant for numerical stability                  |

---

## ✅ Results

### Fine-tuning Performance

| Dataset          | No Finetune | Finetune | Finetune + Hyperparameter Tuning |
| ---------------- | ----------- | -------- | -------------------------------- |
| IMDb             | 49.9%       | 93.2%    | 93.2%                            |
| dair-ai/emotion  | 34.9%       | 92.4%    | 92.5%                            |
| Sp1786 sentiment | 37.0%       | 74.9%    | 75.0%                            |

### 🕒 Training Time

| Setting                | Time   |
| ---------------------- | ------ |
| Baseline (fp32)        | 624.2s   |
| fp16 (mixed precision) | 621.8s |

> 📝 Mixed precision training yielded minimal improvement due to the small model and dataset size.

---

## 🔄 Generalization Study

When the IMDb-finetuned model is directly evaluated on other datasets:

| Target Dataset   | Accuracy |
| ---------------- | -------- |
| dair-ai/emotion  | 18.0%    |
| Sp1786 sentiment | 24.9%    |

> Finetuned models on IMDb do **not generalize well** to emotion or multiclass sentiment datasets, highlighting task/domain specificity.
