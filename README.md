Goal: fine tune a sentiment analysis model

## TODO
* 其他 DATASETS
    * finetune 後與 finetune
    * 不同模型的交互
* 更改模型後的形狀
* 了解 huggingface trainer 能修改的參數
    * performance: activation、model_arch
    * efficiency: quantization 

## Dataset
>IMDb Large Movie Review Dataset: downloads from huggingface
>#data: 50K (movie review, negative/positive)
>type: binary sentiment classification

>dair-ai/emotion

>Sp1786/multiclass-sentiment-analysis-dataset

## Pretrained - Models
>bert-base-uncased: downloads from huggingface
>>>wiki + bookCorpus: 訓練成的

### Traing Arguments
### **Optimizer & Learning Rate**
* `learning_rate`: initial learning rate
* `weight_decay`: L2 regularization (used correctly only with AdamW)
* `lr_scheduler_type`: type of scheduler
  * options: `"linear"`, `"cosine"`, `"polynomial"`, etc.
* `warmup_steps`: number of warmup steps for LR scheduler
* `adam_beta1`, `adam_beta2`, `adam_epsilon`: optional fine-tuning of Adam optimizer behavior


## Results
IMDB
with finetune+hyperparameter: 93.2%
with finetune: 93.2%
without finetune: 49.9%

dair-ai/emotion
with finetune+hyperparameter: 92.5%
with finetune: 92.4%
without finetune: 34.9%

Sp1786
with finetune+hyperparameter: 75%
with finetune: 75%
without finetune: 37.0%

### Generalization
IMDB to dair-ai/emotion and Sp1786

## Learning
命名邏輯
* utils
    * 動作_utils.py 
* scripts
    * 動作.py
* experiments
    * 動作_模型_資料集[_補充說明].sh
