cd "$(dirname "$0")"/../..
python3 evaluate.py \
--model results/imdb_finetune_hyper/run-0/checkpoint-626 \
--dataset Sp1786/multiclass-sentiment-analysis-dataset \
--text_col text \
--label_col label \
--n_labels 6 \
--output_dir results \