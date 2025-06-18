from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

def set_eval_agent(model_name, num_labels, metrics_strategy):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)

    args = TrainingArguments(output_dir="./results", per_device_eval_batch_size=16)

    eval_agent = Trainer(
        model=model,
        args=args,
        compute_metrics=metrics_strategy,
    )
    return eval_agent

def set_train_agent(model_name, train_dataset, eval_dataset, tokenizer, num_labels, metrics_strategy, output_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    train_agent = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=metrics_strategy,
    )
    return train_agent

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }

def run_train_agent_hyper(model_name, train_dataset, eval_dataset, tokenizer, num_labels, metrics_strategy, output_dir, log_dir):
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=log_dir,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        disable_tqdm=False
    )

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    train_agent = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=metrics_strategy,
    )

    best_run = train_agent.hyperparameter_search(
        direction="maximize",
        hp_space=hp_space,
        n_trials=2,
        compute_objective=lambda metrics: metrics["eval_accuracy"],
        backend="optuna"
    )

    return best_run

