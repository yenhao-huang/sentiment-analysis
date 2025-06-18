cd "$(dirname "$0")"/..
python3 finetune_hyper.py \
--model bert-base-uncased \
--dataset Sp1786/multiclass-sentiment-analysis-dataset \
--text_col text \
--label_col label \
--n_labels 3 \
--output_dir results \
