import argparse
from utils import utils, data_utils, model_utils, metric_utils, log_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--n_labels", type=int, default=2)
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_dir", type=str, default="./logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir, log_dir = utils.make_dir_with_timestamp(args.output_dir, args.log_dir)
    raw_data = data_utils.data_loading(args.dataset)
    tonkenized_data, tokenizer = data_utils.data_preprocess(raw_data, args.model, args.text_col, args.label_col)
    train_dataset, eval_dataset = data_utils.select_partial_data(tonkenized_data, 10000, 2000)
    best_run = model_utils.run_train_agent_hyper(
        args.model, 
        train_dataset, 
        eval_dataset, 
        tokenizer, 
        args.n_labels, 
        metric_utils.compute_metrics,
        output_dir,
        log_dir,
    )
    log_utils.log_best_trail_hyper(best_run, output_dir)