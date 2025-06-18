cd "$(dirname "$0")"/..
python3 evaluate.py \
--model results/imdb_finetune/checkpoint-1250 \
--dataset imdb \
--text_col text \
--label_col label \
--n_labels 2 \
--output_dir results \