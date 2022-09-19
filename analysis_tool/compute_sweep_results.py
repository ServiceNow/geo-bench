"""Compute sweep results."""

import argparse
import glob
import math
import os
import pickle
from datetime import datetime
from email.quoprimime import unquote
from pathlib import Path

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from tqdm import tqdm


def retrieve_runs(args):
    """Compute results for a sweep."""
    sweep_exps = glob.glob(os.path.join(args.experiment_dir, "**", "**", "csv_logs", "**", "config.yaml"))

    csv_run_dirs = [Path(path).parent for path in sweep_exps]
    all_trials_df = pd.DataFrame(
        columns=[
            "train_loss",
            "train_metric",
            "val_loss",
            "val_metric",
            "test_loss",
            "test_metric",
            # "step",
            "epoch",
            "unix_time",
            "dataset",
            "model",
            "partition_name",
            "benchmark_name",
            "batch_size",
            "exp_dir",
            "csv_log_dir",
        ]
    )

    if args.existing_csv:
        exist_df = pd.read_csv(args.existing_csv)
        exist_csv_dirs = exist_df["csv_log_dir"].unique().tolist()
    skip_counter = 0
    for csv_logger_dir in tqdm(csv_run_dirs):
        if args.existing_csv:
            if str(csv_logger_dir) in exist_csv_dirs:
                continue

        # load task_specs
        with open(csv_logger_dir.parents[1] / "task_specs.pkl", "rb") as f:
            task_specs = pickle.load(f)

        # load config
        yaml = YAML()
        with open(csv_logger_dir / "config.yaml", "r") as fd:
            config = yaml.load(fd)

        # load metric_csv
        orig_df = pd.read_csv(csv_logger_dir / "metrics.csv")
        # keep track of steps to epoch and use trick to remove stacked nans
        step_to_epoch = orig_df[["step", "epoch"]].drop_duplicates(subset="step")
        step_to_epoch = dict(zip(step_to_epoch.step, step_to_epoch.epoch))

        eval_df = orig_df[orig_df["val_loss"].notnull()]
        eval_df = eval_df.groupby("epoch").mean().reset_index()
        train_loss_df = orig_df[orig_df["train_loss"].notnull()]
        train_loss_df = train_loss_df.groupby("epoch").mean().reset_index()

        metric_name = str(task_specs.eval_loss).split(".")[-1].split("'")[0]
        if metric_name == "MultilabelAccuracy":
            metric_name = "F1Score"

        if metric_name == "SegmentationAccuracy":
            metric_name = "JaccardIndex"

        if metric_name == "Accuracy" and "NeonTree_segmentation" in str(csv_logger_dir):
            metric_name = "JaccardIndex"

        train_metric_df = orig_df[orig_df["train_" + metric_name].notnull()]

        train_metric_df = train_metric_df.groupby("epoch").mean().reset_index()

        eval_df.update(train_loss_df)
        eval_df.update(train_metric_df)

        step_to_time = eval_df[["step", "current_time"]].drop_duplicates(subset="step")
        step_to_time = dict(zip(step_to_time.step, step_to_time.current_time))

        metric_name = metric_name.lower()

        # new column names to be universal as specific metric name might differ with tasks
        new_columns = []
        for col in eval_df.columns:
            if metric_name in col.lower():
                new_name = col.lower().split("-")[0].replace(metric_name, "metric")
            else:
                new_name = col
            new_columns.append(new_name)

        eval_df = eval_df.rename(columns={old: new for old, new in zip(eval_df.columns, new_columns)})

        # not finished runs
        if not all(
            x in eval_df.columns
            for x in ["train_loss", "val_loss", "test_loss", "train_metric", "val_metric", "test_metric"]
        ):
            skip_counter += 1
            continue

        best_val_train = eval_df.iloc[eval_df["val_loss"].argmin(), :][
            [
                "epoch",
                "current_time",
                "train_loss",
                "train_metric",
                "val_loss",
                "val_metric",
                "test_loss",
                "test_metric",
            ]
        ]
        # best_test = eval_df.iloc[len(eval_df) - 1, :][["test_loss", "test_metric"]]

        if os.path.basename(config["experiment"]["benchmark_dir"]).startswith("segmentation"):
            model = config["model"]["encoder_type"] + "_" + config["model"]["decoder_type"]
        else:
            model = config["model"]["backbone"]

        avg_epoch_time = eval_df["current_time"].diff().mean()
        compute_time = avg_epoch_time * best_val_train.epoch

        best_scores = [
            [
                best_val_train.train_loss,
                best_val_train.train_metric,
                best_val_train.val_loss,
                best_val_train.val_metric,
                best_val_train.test_loss,
                best_val_train.test_metric,
                best_val_train.epoch,
                compute_time,
                task_specs.dataset_name,
                model,
                os.path.basename(config["experiment"]["partition_name"]),
                os.path.basename(config["experiment"]["benchmark_dir"]),
                config["model"]["batch_size"],
                str(csv_logger_dir.parents[1]),
                str(csv_logger_dir),
            ]
        ]

        all_trials_df = pd.concat([all_trials_df, pd.DataFrame(best_scores, columns=all_trials_df.columns)])

    all_trials_df.reset_index(drop=True, inplace=True)

    # remove duplicates sweeps and keep the ones with 12 trials
    count_df = all_trials_df.groupby(["model", "dataset", "partition_name", "exp_dir"]).size().reset_index()
    count_df.rename(columns={0: "count"}, inplace=True)

    # extract latest date from string
    count_df["date"] = count_df["exp_dir"].str.split("/", 0, expand=True)[6].str.split("_", 4, expand=True)[4]
    count_df["date"] = pd.to_datetime(count_df["date"], format="%m-%d-%Y_%H:%M:%S")

    count_df.sort_values(by=["model", "dataset", "partition_name", "exp_dir", "date"], inplace=True, ascending=False)
    # drop if there are duplicates with equal count
    count_df.drop_duplicates(subset=["model", "dataset", "partition_name"], inplace=True, keep="first")
    exp_dirs_to_keep = count_df["exp_dir"].tolist()
    all_trials_df = all_trials_df[all_trials_df["exp_dir"].isin(exp_dirs_to_keep)]

    if args.existing_csv:
        all_trials_df = pd.concat([all_trials_df, exist_df])
        # remove duplicates sweeps and keep the ones with 12 trials
        count_df = all_trials_df.groupby(["model", "dataset", "partition_name", "exp_dir"]).size().reset_index()
        count_df.rename(columns={0: "count"}, inplace=True)
        # count_df = count_df[count_df["count"] == 12]
        # extract latest date from string
        count_df["date"] = count_df["exp_dir"].str.split("/", 0, expand=True)[6].str.split("_", 4, expand=True)[4]
        count_df["date"] = pd.to_datetime(count_df["date"], format="%m-%d-%Y_%H:%M:%S")

        count_df.sort_values(
            by=["model", "dataset", "partition_name", "exp_dir", "date"], inplace=True, ascending=False
        )
        count_df.drop_duplicates(subset=["model", "dataset", "partition_name"], inplace=True, keep="first")
        exp_dirs_to_keep = count_df["exp_dir"].tolist()
        all_trials_df = all_trials_df[all_trials_df["exp_dir"].isin(exp_dirs_to_keep)]

    # save new eval_df
    all_trials_df.to_csv(
        os.path.join(args.result_dir, f"sweep_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"), index=False
    )


def start():
    """Start function."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="compute_sweep_results.py",
        description="Generate experiment directory structure based on user-defined model generator",
    )
    parser.add_argument(
        "--experiment_dir",
        help="The based directory in which experiment-related files should be searched and included in report.",
        required=True,
    )
    parser.add_argument(
        "--existing_csv",
        help="Path to existing summary table to which to append new results.",
    )

    parser.add_argument("--result_dir", help="Directory where resulting overview should be saved.", required=True)

    args = parser.parse_args()

    retrieve_runs(args)


if __name__ == "__main__":
    start()
