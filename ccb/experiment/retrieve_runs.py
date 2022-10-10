"""Retrieve runs from experiment directory for analysis."""

import glob
import math
import os
import pickle
from datetime import datetime
from email.quoprimime import unquote
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from ruamel.yaml import YAML
from tqdm import tqdm


def retrieve_runs(sweep_experiment_dir, use_cached_csv=False, exp_type="sweep"):
    """Compute results for a sweep.

    Args:
        experiment_dir: diretory where all sweeps are stored
        use_cached_csv: cached csv
        exp_type: 'sweep', 'seeds'

    Returns:
        df with sweep summaries per individual run
    """
    csv_path = Path(sweep_experiment_dir) / "cached_results.csv"
    if use_cached_csv and csv_path.exists():
        return pd.read_csv(csv_path)

    if exp_type == "sweep":
        run_dirs = glob.glob(os.path.join(sweep_experiment_dir, "**", "**", "csv_logs", "**", "config.yaml"))
    elif exp_type == "seeds":
        run_dirs = glob.glob(os.path.join(sweep_experiment_dir, "**", "**", "**", "csv_logs", "**", "config.yaml"))
    else:
        raise ValueError("exp_type not valid, should be 'sweep' or 'seeds'")

    csv_run_dirs = [Path(path).parent for path in run_dirs]
    all_trials_df = pd.DataFrame(
        columns=[
            "train_loss",
            "train_metric",
            "val_loss",
            "val_metric",
            "test_loss",
            "test_metric",
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

    for csv_logger_dir in tqdm(csv_run_dirs):

        # load task_specs
        with open(csv_logger_dir.parents[1] / "task_specs.pkl", "rb") as f:
            task_specs = pickle.load(f)

        # load config
        yaml = YAML()
        with open(csv_logger_dir / "config.yaml", "r") as fd:
            config = yaml.load(fd)

        # load metric_csv
        if os.path.exists(os.path.join(str(csv_logger_dir), "metrics.csv")):
            try:
                orig_df = pd.read_csv(csv_logger_dir / "metrics.csv")
            except EmptyDataError:
                continue
        else:
            continue
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

        if "segmentation" in config["model"]["model_generator_module_name"]:
            model = config["model"]["encoder_type"] + "_" + config["model"]["decoder_type"]
        elif "ssl_moco" in config["model"]["model_generator_module_name"]:
            model = "ssl_moco_" + config["model"]["backbone"]
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

    all_trials_df["date"] = (
        all_trials_df["exp_dir"].str.split("_", expand=True)[6]
        + "_"
        + all_trials_df["exp_dir"].str.split("_", expand=True)[7]
    )

    most_recent = all_trials_df.groupby(["model", "dataset", "partition_name"]).head(12)

    # # remove duplicates sweeps and keep the ones with 12 trials
    # count_df = all_trials_df.groupby(["model", "dataset", "partition_name", "exp_dir"]).size().reset_index()
    # count_df.rename(columns={0: "count"}, inplace=True)

    # # extract latest date from string
    # count_df["date"] = (
    #     count_df["exp_dir"].str.split("_", expand=True)[6] + "_" + count_df["exp_dir"].str.split("_", expand=True)[7]
    # )
    # count_df["date"] = pd.to_datetime(count_df["date"], format="%m-%d-%Y_%H:%M:%S")

    # # keep the most recent version
    # import pdb
    # pdb.set_trace()
    # count_df.sort_values(by=["model", "dataset", "partition_name", "exp_dir", "date"], inplace=True, ascending=False)
    # if exp_type == "sweep":
    #     count_df.drop_duplicates(subset=["model", "dataset", "partition_name"], inplace=True, keep="first")
    # exp_dirs_to_keep = count_df["exp_dir"].tolist()
    # all_trials_df = all_trials_df[all_trials_df["exp_dir"].isin(exp_dirs_to_keep)].reset_index(drop=True)

    all_trials_df.to_csv(csv_path)

    return most_recent


partition_names = [
    "0.01x_train",
    "0.02x_train",
    "0.05x_train",
    "0.10x_train",
    "0.20x_train",
    "0.50x_train",
    "1.00x_train",
    "default",
]
classification_dataset_names = [
    "bigearthnet",
    "brick_kiln_v1.0",
    "eurosat",
    "pv4ger_classification",
    "so2sat",
    "forestnet_v1.0",
    "geolifeclef-2022",
]

classification_models = [
    "conv4",
    "resnet18",
    "resnet50",
    "convnext_base",
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "swinv2_tiny_window16_256",
    "ssl_moco_resnet18",
    "ssl_moco_resnet50",
]

segmentation_dataset_names = [
    "pv4ger_segmentation",
    "nz_cattle_segmentation",
    "smallholder_cashew",
    "southAfricaCropType",
    "cvpr_chesapeake_landcover",
]

segmentation_models = [
    "resnet18_Unet",
    "resnet50_Unet",
    "resnet101_Unet",
    "resnet18_DeepLabV3",
    "resnet50_DeepLabV3",
    "resnet101_DeepLabV3",
]


def find_missing_runs(df, num_run_thresh: int = 10, task: str = "classification"):
    """Find missing runs for dataset and model combinations.

    Args:
        df: dataframe results of above retrieve runs function
        num_run_thresh: number of runs to consider as a combination to be not complete
        task: segmentation or classification

    Returns:
        dict that shows missing runs
    """
    if task == "classification":
        dataset_names = classification_dataset_names
    elif task == "segmentation":
        dataset_names = segmentation_dataset_names
    else:
        print("Task is not defined.")

    # each model and dataset and partition is present:
    miss_dict: Dict[str, Dict[str, List]] = {}
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        miss_dict[model] = {}
        for part in partition_names:
            part_df = model_df[model_df["partition_name"] == part]
            miss_dict[model][part] = []
            for ds in dataset_names:
                ds_df = part_df[part_df["dataset"] == ds]
                exp_dirs = ds_df["exp_dir"].unique()
                if len(exp_dirs) == 0:
                    # miss_dict[model][part].append(ds)
                    continue
                else:
                    for exp_dir in exp_dirs:
                        exp_df = ds_df[ds_df["exp_dir"] == exp_dir]
                        if len(exp_df) < num_run_thresh:
                            miss_dict[model][part].append(ds)
    return miss_dict


# retrieve_runs("/mnt/data/experiments/nils/new_classification_seeded_runs")
