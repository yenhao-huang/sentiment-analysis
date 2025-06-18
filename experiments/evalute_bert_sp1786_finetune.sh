cd "$(dirname "$0")"/..
python3 evaluate.py \
--model  results/sp1786_finetune/checkpoint-1250/ \
--dataset Sp1786/multiclass-sentiment-analysis-dataset \
--text_col text \
--label_col label \
--n_labels 3 \
--output_dir results \