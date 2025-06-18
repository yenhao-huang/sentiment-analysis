cd "$(dirname "$0")"/..
python3 evaluate.py \
--model  bert-base-uncased \
--dataset dair-ai/emotion \
--text_col text \
--label_col label \
--n_labels 6 \
--output_dir results \