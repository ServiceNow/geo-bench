import argparse
import glob
import os
import json
import yaml
import pickle
import pandas as pd
import wandb


def create_overview(args):
    """Provide an overview of which experiments and sweeps have already been conducted.

    Args:
        args: parsed command line arguments
    """
    # check if metrics files are present to only include sweeps that have been successfully run
    sweep_exps = glob.glob(os.path.join(args.experiment_dir, "**", "**", "lightning_logs", "**", "metrics.csv"))
    sweep_exps = list(set(["/".join(path.split("/")[:-3]) for path in sweep_exps]))

    cols = [
        "benchmark_name",
        "dataset_name",
        "backbone",
        "loss_function",
        "metric",
        "hyperparameter",
        "best_value",
        "test_loss",
        "test_metric",
        "train_loss",
        "train_metric",
        "val_loss",
        "val_metric",
        "batch_size",
        "model_generator_name",
        "job_dir",
        "sweep_id",
    ]
    overview_df = pd.DataFrame(columns=cols)

    for exp_path in sweep_exps:
        # load hparams
        with open(os.path.join(exp_path, "hparams.json"), "r") as f:
            hparams = json.load(f)

        # load task_specs
        with open(os.path.join(exp_path, "task_specs.pkl"), "rb") as f:
            task_specs = pickle.load(f)

        api = wandb.Api()
        if "sweep_id" in hparams:
            sweep_id = hparams["sweep_id"]
        else:
            continue

        # if sweep on wandb was deleted
        try:
            sweep = api.sweep(sweep_id)
        except wandb.errors.CommError:
            continue

        best_run = sweep.best_run()

        if best_run is None:
            continue

        wandb_config = best_run.config
        wandb_summary = best_run.summary

        # if run has not finished properly
        if "test_loss" not in wandb_summary:
            continue

        # filter wandb_summary to only include metrics start with train, val, or test
        loss_function = hparams["loss_type"]
        metric_name = str(task_specs.eval_loss).split(".")[-1].split("'")[0].lower()
        sweep_summary = {}
        for key, value in wandb_summary.items():
            if key in ["Accuracy", "F1Score"]:
                key = "train_" + key
            if key.startswith(("train_", "val_", "test_")):
                if metric_name in key.lower():
                    new_key = key.lower().replace(metric_name, "metric")
                else:
                    new_key = key
                sweep_summary[new_key] = value

        sweep_summary = {key: value for key, value in sorted(sweep_summary.items())}

        # load sweep_config
        with open(os.path.join(exp_path, "sweep_config.yaml"), "r") as f:
            sweep_config = yaml.safe_load(f)

        swept_params = sweep_config["parameters"]
        param_names = list(swept_params.keys())

        param_vals = {}
        for param in param_names:
            param_vals[param] = wandb_config[param]

        benchmark_name = task_specs.benchmark_name
        dataset_name = task_specs.dataset_name
        backbone = hparams["backbone"]
        model_generator_name = hparams["model_generator_name"]
        batch_size = hparams["batch_size"]

        df_rows = []

        for param_name, param_val in param_vals.items():
            df_rows.append(
                [benchmark_name, dataset_name, backbone, loss_function, metric_name, param_name, param_val]
                + list(sweep_summary.values())
                + [batch_size, model_generator_name, exp_path, sweep_id]
            )

        exp_df = pd.DataFrame(df_rows, columns=cols)

        overview_df = overview_df.append(exp_df, ignore_index=True)

    overview_df.sort_values(
        by=["benchmark_name", "dataset_name", "backbone", "sweep_id", "hyperparameter"], axis=0, inplace=True
    )
    overview_df.reset_index(drop=True, inplace=True)

    overview_df.to_csv(os.path.join(args.result_dir, "sweep_overview.csv"), index=False)

    # create a "best" hyperparameter file that for each backbone has the best hyperparams and
    # can be used to run models from multiple seeds
    best_df = (
        overview_df.groupby(["dataset_name", "backbone"])
        .apply(lambda group: group.loc[group["val_loss"] == group["val_loss"].min()])
        .reset_index(level=-1, drop=True)
    )

    # per dataset for each backbone extract the applied hyperparameter setting
    best_hyperparams_summary = dict()

    dataset_names = best_df["dataset_name"].unique()
    for ds_name in dataset_names:
        ds_df = best_df[best_df["dataset_name"] == ds_name]

        best_hyperparams_summary[ds_name] = {}
        backbone_names = ds_df["backbone"].unique()

        for back_name in backbone_names:
            back_df = ds_df[ds_df["backbone"] == back_name].reset_index(drop=True)

            # collect optimal hyperparameters for this
            job_dir = back_df.loc[0, "job_dir"]
            benchmark = back_df.loc[0, "benchmark_name"]
            dataset = back_df.loc[0, "dataset_name"]
            model_generator_name = back_df.loc[0, "model_generator_name"]
            val_loss = back_df.loc[0, "val_loss"]
            val_metric = back_df.loc[0, "val_metric"]
            batch_size = back_df.loc[0, "batch_size"]

            with open(os.path.join(job_dir, "hparams.json"), "r") as f:
                hparams = json.load(f)

            # get tuned hyperparameter and values
            tuned_params = back_df["hyperparameter"].tolist()
            tuned_vals = back_df["best_value"].tolist()

            # overwrite params
            for name, val in zip(tuned_params, tuned_vals):
                hparams[name] = val

            hparams["batch_size"] = int(batch_size)
            hparams["hidden_size"] = int(hparams["hidden_size"])
            hparams["dataset_name"] = dataset
            hparams["benchmark_name"] = benchmark
            hparams["model_generator_name"] = model_generator_name
            hparams["val_loss"] = val_loss
            hparams["val_metric"] = val_metric

            best_hyperparams_summary[ds_name][back_name] = hparams

    with open(os.path.join(args.result_dir, "best_hparams_found.json"), "w") as f:
        json.dump(best_hyperparams_summary, f)

    print("Done")


def start():
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="experiment_overview.py",
        description="Generate experiment directory structure based on user-defined model generator",
    )
    parser.add_argument(
        "--experiment_dir",
        help="The based directory in which experiment-related files should be searched and included in report.",
        required=True,
    )

    parser.add_argument(
        "--result_dir",
        help="Directory where resulting overview should be saved.",
        required=True,
    )

    args = parser.parse_args()

    create_overview(args)


if __name__ == "__main__":
    start()
