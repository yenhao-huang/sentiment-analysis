cd "$(dirname "$0")"/..
python3 evaluate.py \
--model results/dairai_finetune/checkpoint-1250 \
--dataset dair-ai/emotion \
--text_col text \
--label_col label \
--n_labels 6 \
--output_dir results \