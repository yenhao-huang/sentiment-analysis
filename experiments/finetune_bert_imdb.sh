cd "$(dirname "$0")"/..
python3 finetune.py \
--model bert-base-uncased \
--dataset imdb \
--text_col text \
--label_col label \
--n_labels 2 \
--output_dir results \
