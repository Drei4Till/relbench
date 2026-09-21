import argparse
import os
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd

from relbench import load_dataset
from relbench.base import Dataset, RecommendationTask, Table
from relbench.submit import evaluate_task, write_prediction_table

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stack")
parser.add_argument("--task", type=str, default="user-post-comment")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--pred_dir", type=str, default="/tmp/relbench_preds")
args = parser.parse_args()


np.random.seed(args.seed)

dataset: Dataset = load_dataset(args.dataset)
task: RecommendationTask = dataset.load_task(args.task)

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")

trainval_table_df = pd.concat([train_table.df, val_table.df], axis=0)
trainval_table = Table(
    df=trainval_table_df,
    fkey_col_to_pkey_table=train_table.fkey_col_to_pkey_table,
    pkey_col=train_table.pkey_col,
    time_col=train_table.time_col,
)


def past_visit_aggr(x):
    lst_cat = []
    for e in list(x):
        lst_cat.extend(e)
    counter = Counter(lst_cat)
    topk = [elem for elem, _ in counter.most_common(task.eval_k)]
    # padding
    if len(topk) < task.eval_k:
        topk.extend([-1] * (task.eval_k - len(topk)))
    return topk


def predict(
    train_table: Table,
    pred_table: Table,
    name: str,
) -> np.ndarray:
    if name == "past_visit":
        """Predict the most frequently-visited dst nodes per each src node."""
        df = (
            train_table.df.groupby(task.src_entity_col)[task.dst_entity_col]
            .apply(past_visit_aggr)
            .reset_index(name="__pred__")
        )
        pred_ser = pd.merge(pred_table.df, df, how="left", on=task.src_entity_col)[
            "__pred__"
        ]
        # Replace NaN with [-1, -1, ..., -1] prediction
        pred_ser = pred_ser.apply(
            lambda x: x if isinstance(x, list) else [-1] * task.eval_k
        )
        pred = np.stack(pred_ser.values)
    elif name == "global_popularity":
        """Predict the globally most visited dst nodes and predict them across the src
        nodes."""
        lst_cat = []
        for lst in train_table.df[task.dst_entity_col]:
            lst_cat.extend(lst)
        counter = Counter(lst_cat)
        topk = [elem for elem, _ in counter.most_common(task.eval_k)]
        # padding
        if len(topk) < task.eval_k:
            topk.extend([-1] * (task.eval_k - len(topk)))
        pred = np.tile(np.array(topk), (len(pred_table), 1))
    else:
        raise ValueError(f"Unknown eval name called {name}.")
    return pred


def evaluate(
    train_table: Table,
    pred_table: Table,
    name: str,
) -> Dict[str, float]:
    pred = predict(train_table, pred_table, name)
    return task.evaluate(pred, pred_table)

metrics_dict = {}

eval_name_list = ["past_visit", "global_popularity"]
for name in eval_name_list:
    train_metrics = evaluate(train_table, train_table, name=name)
    val_metrics = evaluate(train_table, val_table, name=name)
    test_pred = predict(trainval_table, test_table, name=name)
    os.makedirs(args.pred_dir, exist_ok=True)
    pred_path = os.path.join(args.pred_dir, f"{args.dataset}__{args.task}.csv")
    write_prediction_table(task, test_pred, pred_path)
    test_metrics = evaluate_task(f"{args.dataset}/{args.task}", pred_path)
    print(f"{name}:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")
    print(f"Test: {test_metrics}")
    metrics_dict[name] = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }


import json 

output_path = os.path.join("results", args.dataset, args.task)
os.makedirs(output_path, exist_ok=True)

slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
file_path = os.path.join(output_path, str(args.seed) + "_trivial_" + str(slurm_job_id) + ".json")
with open(file_path, "w") as f:
    json.dump(metrics_dict, f, indent=4)

print(f"Trivial predictions complete. You may look for the results under: {output_path}")