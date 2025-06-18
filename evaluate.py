import argparse
from utils import data_utils, model_utils, metric_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--n_labels", type=int, default=2)
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw_data = data_utils.data_loading(args.dataset)
    tonkenized_data, _ = data_utils.data_preprocess(raw_data, args.model, args.text_col, args.label_col)
    eval_agent = model_utils.set_eval_agent(args.model, args.n_labels, metric_utils.compute_metrics)
    results = eval_agent.evaluate(eval_dataset=tonkenized_data["test"])
    print("Accuracy:", results["eval_accuracy"])
