"""Parse results."""

from collections import defaultdict
from functools import cache
from pathlib import Path
from textwrap import wrap
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import trim_mean

from ccb.dataset_converters import inspect_tools


def make_normalizer(data_frame, metrics=("test metric",)):
    """Extract min and max from data_frame to build Normalizer object for all datasets."""
    datasets = data_frame["dataset"].unique()
    range_dict = {}

    for dataset in datasets:
        sub_df = data_frame[data_frame["dataset"] == dataset]
        data = []
        for metric in metrics:
            data.append(sub_df[metric].to_numpy())
        range_dict[dataset] = (np.min(data), np.max(data))

    return Normalizer(range_dict)


class Normalizer:
    """Class used to normalize results beween min and max for each dataset."""

    def __init__(self, range_dict):
        """Initialize a new instance of Normalizer class."""
        self.range_dict = range_dict

    def __call__(self, ds_name, values, scale_only=False):
        """Call the Normalizer class."""
        mn, mx = self.range_dict[ds_name]
        range = mx - mn
        if scale_only:
            return values / range
        else:
            return (values - mn) / range

    def from_row(self, row, scale_only=False):
        """Normalize from row."""
        return [self(ds_name, val, scale_only=scale_only) for ds_name, val in row.items()]

    def normalize_data_frame(self, df, metrics):
        """Normalize the entire dataframe."""
        for metric in metrics:
            new_metric = f"normalized {metric}"
            df[new_metric] = df.apply(lambda row: self.__call__(row["dataset"], row[metric]), axis=1)

def biqm(scores):
    """Return a bootstram sample of iqm."""
    b_scores = np.random.choice(scores, size=len(scores), replace=True)
    return trim_mean(b_scores, proportiontocut=0.25, axis=None)


def iqm(scores):
    """Interquantile mean."""
    return trim_mean(scores, proportiontocut=0.25, axis=None)


def bootstrap_iqm(df, group_keys=("model", "dataset"), metric="test_metric", repeat=100):
    """Boostram of seeds for all model and all datasets to comput iqm score distribution."""
    df_list = []
    for i in range(repeat):
        series = df.groupby(list(group_keys))[metric].apply(biqm)
        df_list.append(series.to_frame().reset_index())

    return pd.concat(df_list)


def bootstrap_iqm_aggregate(df, metric="test_metric", repeat=100):
    """Stratified bootstrap (by dataset) of all seeds to compute iqm score distribution for each model."""
    group = df.groupby(["dataset", "model"])
    df_list = []
    for i in range(repeat):
        new_df = group.sample(frac=1, replace=True)
        series = new_df.groupby(["model"])[metric].apply(iqm)
        df_list.append(series.to_frame().reset_index())

    new_df = pd.concat(df_list)
    new_df.loc[:, "dataset"] = "aggregated"
    return new_df


def plot_bootstrap_aggregate(df, metric, model_order, repeat=100, fig_size=None):
    """Add aggregated data as a new dataset."""
    bootstrapped_iqm = pd.concat(
        (
            bootstrap_iqm_aggregate(df, metric=metric, repeat=repeat),
            bootstrap_iqm(df, metric=metric, repeat=repeat),
        )
    )
    plot_per_dataset_3(bootstrapped_iqm, model_order, metric=metric, fig_size=fig_size)




def plot_per_dataset_3(
    df,
    model_order,
    metric="test metric",
    aggregated_name="aggregated",
    sharey=True,
    inner="box",
    fig_size=None,
    n_legend_rows=1,
):
    """Violin plots for each datasets and each models."""
    datasets = sorted(df["dataset"].unique())
    if fig_size is None:
        fig_width = len(datasets) * 2
        fig_size = (fig_width, 3)
    fig, axes = plt.subplots(1, len(datasets), sharey=sharey, figsize=fig_size)

    for dataset, ax in zip(datasets, axes):
        sns.set_style("dark")

        sub_df = df[df["dataset"] == dataset]
        sns.violinplot(
            x="dataset",
            y=metric,
            hue="model",
            data=sub_df,
            hue_order=model_order,
            linewidth=0.1,
            saturation=0.9,
            scale="count",
            inner=inner,
            ax=ax,
        )
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        if dataset == aggregated_name:
            ax.set_facecolor("#cff6fc")

        ax.set(xlabel=None)

        if dataset != datasets[int((len(datasets) - 1) / 2)]:
            ax.get_legend().remove()
        else:
            ncols = int(np.ceil(len(model_order) / n_legend_rows))
            sns.move_legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=ncols, title="")

        if dataset != datasets[0]:
            if sharey:
                ax.axes.get_yaxis().set_visible(False)
            else:
                ax.set(ylabel=None)

    if sharey:
        fig.subplots_adjust(wspace=0.01)
    else:
        fig.subplots_adjust(wspace=0.3)

def plot_normalized_time(df: pd.DataFrame, time_metric="step", average_seeds=True):
    """Plot the time (in number of steps) of each experiment as a function of the training set size."""
    df["train ratio"] = [get_train_ratio(part_name) for part_name in df["partition name"]]

    if time_metric == "step":
        df["n observation"] = df["step"] * df["batch size"]
        time_metric = "n observation"

    mean_1x_time = df[df["train ratio"] == 0.1].groupby(["dataset", "model"])[time_metric].mean()
    df["mean 1x time"] = df.apply(lambda row: mean_1x_time[(row.dataset, row.model)], axis=1)
    normalized_name = f"{time_metric} normalized"
    df[normalized_name] = df.apply(lambda row: row[time_metric] / row["mean 1x time"], axis=1)

    if average_seeds:
        df = df.groupby(["dataset", "train ratio"]).mean()
        df = df.reset_index()
        noise_level = 0.05
    else:
        noise_level = 0.1

    df["train ratio +noise"] = df.apply(
        lambda row: row["train ratio"] * np.exp(np.random.randn() * noise_level), axis=1
    )
    # print(df.keys())
    ax = sns.scatterplot(data=df, x="train ratio +noise", y=normalized_name, hue="dataset")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Normalized Training Time")
    ax.set_xlabel("Train Size Ratio")




def get_train_ratio(part_name):
    """Parse train ratio from partition name."""
    return float(part_name.split("x ")[0])


def clean_names(val):
    """Rename a few strings for more compact formatting."""
    val = inspect_tools.DISPLAY_NAMES.get(val, val)
    if isinstance(val, str):
        for src, tgt in (
            ("_classification", ""),
            ("vit_small_patch16_224", "vit_small"),
            ("vit_tiny_patch16_224", "vit_tiny"),
            ("swinv2_tiny_window16_256", "swinv2"),
            ("_", " "),
        ):
            val = val.replace(src, tgt)

    return val



@cache
def collect_trace_info(log_dir: str, index="step") -> pd.DataFrame:
    """Collect trace infor for a single run.

    Args:
        log_dir: logging directory where trace csv can be found

    Returns:
        dataframe with trace information
    """
    # log_dir = log_dir.replace("train_v0.5_", "train_classification_v0.5_")
    # csv_path = Path(log_dir) / "lightning_logs" / "version_0" / "metrics.csv"

    try:
        df = pd.read_csv(Path(log_dir) / "metrics.csv")
    except pd.EmptyDataError:
        print(f"Empty log: {Path(log_dir) / 'metrics.csv'}")
        return {}

    df.set_index(index)

    trace_dict = {}
    for col_name, series in df.items():
        trace_dict[col_name] = series.dropna()

    return trace_dict


def collect_trace_info_raw(log_dir: str) -> pd.DataFrame:
    """Collect trace infor for a single run.

    Args:
        log_dir: logging directory where trace csv can be found

    Returns:
        dataframe with trace information
    """
    log_dir = log_dir.replace("train_v0.5_", "train_classification_v0.5_")
    csv_path = Path(log_dir) / "lightning_logs" / "version_0" / "metrics.csv"
    df = pd.read_csv(csv_path)
    # df = df.set_index("step").stack().to_frame("value")
    return df


def find_metric_names(keys: List[str]):
    """Find metric based on a key.

    Args:
        keys: List of metrics
    """
    val_metric = None
    test_metric = None
    for key in keys:
        if key in ("val_JaccardIndex", "val_Accuracy", "val_F1Score"):
            val_metric = key
        if key in ("test_JaccardIndex", "test_Accuracy", "test_F1Score"):
            test_metric = key
    return val_metric, test_metric


def smooth_series(series, filt_size):
    """Smoothout the series with a triang window of size filt_size."""
    return series.rolling(filt_size, win_type="triang").mean()


def extract_best_points(log_dirs, filt_size=5, lower_is_better=False, val_metric=None, test_metric=None):
    """Find the optimal step on the validation trace for each log_dir, and return info into a dataframe."""
    max_scores = []
    best_points = []
    for log_dir in log_dirs:
        trace_dict = collect_trace_info(log_dir)

        if val_metric is None or test_metric is None:
            val_metric, test_metric = find_metric_names(trace_dict.keys())

        val_trace = smooth_series(trace_dict[val_metric], filt_size)
        test_trace = trace_dict[test_metric]

        if lower_is_better:  # We currently only ue higher is better.
            scores = -1 * val_trace
        else:
            scores = val_trace

        best_step = scores.idxmax()
        max_scores.append(scores[best_step])  # used for sorting

        best_points.append(
            dict(
                log_dir=log_dir,
                best_step=best_step,
                val_metric=val_trace[best_step],
                test_metric=test_trace[best_step],
                best_config=False,
            )
        )

    best_points = pd.DataFrame(best_points)
    best_points = best_points.set_index("log_dir")

    idx = np.argsort(np.array(max_scores))[::-1]
    sorted_log_dirs = np.array(log_dirs)[idx]
    best_points.at[sorted_log_dirs[0], "best_config"] = True

    return best_points, sorted_log_dirs, val_metric, test_metric


@cache
def get_hparams_old(log_dir: Path):
    """Load the hyper parameters from the yaml file."""
    with open(Path(log_dir) / "hparams.yaml", "r") as stream:
        hparams = yaml.safe_load(stream)
    return hparams


@cache
def get_hparams(log_dir: Path):
    """Load the hyper parameters from the yaml file."""
    with open(Path(log_dir) / "config.yaml", "r") as stream:
        hparams = yaml.safe_load(stream)["model"]

    for key in ("input_size", "model_generator_module_name"):
        hparams.pop(key, None)

    return hparams


def format_hparam(key, value):
    """More compact formatting for learning rates."""
    if isinstance(value, float):
        return f"{key}: {value:.6f}"
    else:
        return f"{key}: {value}"


def format_hparams(log_dirs):
    """Format a configuration of hyperparameters. Only variable values are formatted for more compact display."""
    # load all hparams
    all_configs = defaultdict(list)
    for log_dir in log_dirs:
        hparams = get_hparams(log_dir)
        for key, val in hparams.items():
            all_configs[key].append(val)

    # search for constants vs variables
    constants = {}
    variables = []
    for key, vals in all_configs.items():
        if len(np.unique(vals)) == 1:
            constants[key] = vals[0]
        else:
            variables.append(key)

    # format string displaying variable hyperparams
    exp_names = {}
    for log_dir in log_dirs:
        hparams = get_hparams(log_dir)

        exp_names[log_dir] = ", ".join([format_hparam(key, hparams[key]) for key in variables])
    cst_str = str(constants)
    cst_str = "\n".join(wrap(cst_str, 100))
    return cst_str, exp_names


def make_plot_sweep(filt_size=5, top_k=6, legend=False):
    """Return a plotting function."""

    def plot_sweep(df, ax, dataset, model):
        """Display the validation loss and accuracy for the `top_k` experiments."""
        log_dirs = df["csv_log_dir"]

        ax2 = None
        all_val_loss = []
        # all_val_accuarcy = []

        if len(log_dirs) == 0:
            return

        constants, exp_names = format_hparams(log_dirs)
        best_points, sorted_log_dirs, val_metric, test_metric = extract_best_points(log_dirs, filt_size=filt_size)

        # print(f"best config of {model} on {dataset}: \n{log_dirs[0]}")

        colors = sns.color_palette("tab10")
        for i, log_dir in enumerate(sorted_log_dirs[:top_k]):
            trace_dict = collect_trace_info(log_dir)
            val_loss = smooth_series(trace_dict["val_loss"], filt_size)
            val_trace = smooth_series(trace_dict[val_metric], filt_size)
            test_trace = smooth_series(trace_dict[test_metric], filt_size)

            # print(np.min(trace_dict["val_loss"].keys().to_numpy()))
            all_val_loss.append(val_loss.to_numpy())

            label = exp_names[log_dir] if legend else None

            sns.lineplot(data=val_loss, ax=ax, label=label, color=colors[i])
            if ax2 is None:
                ax2 = ax.twinx()
            sns.lineplot(data=val_trace, ax=ax2, linestyle=":", color=colors[i])
            sns.lineplot(data=test_trace, ax=ax2, linestyle="--", color=colors[i])

            best_step = best_points.at[log_dir, "best_step"]
            val_best_value = best_points.at[log_dir, "val_metric"]
            test_value = best_points.at[log_dir, "test_metric"]
            sns.lineplot(x=[best_step], y=[val_best_value], marker="*", markersize="15", color=colors[i])
            sns.lineplot(x=[best_step], y=[test_value], marker="X", markersize="15", color=colors[i])

        # mn, mx = np.nanpercentile(np.concatenate(all_val_loss), q=[0, 99])
        # ax.set_ylim(bottom=mn, top=mx)

        if legend:
            ax.legend()
            sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.2, 1), title=constants)

        return best_points

    return plot_sweep


def plot_all_models_datasets(df, plot_fn=make_plot_sweep(legend=False), fig_size=None):
    """Create a grid plot for all models and datasets."""
    models = df["model"].unique()
    datasets = df["dataset"].unique()

    new_df = pd.DataFrame()
    fig, axes = plt.subplots(len(datasets), len(models), figsize=fig_size)
    # fig.suptitle(metric, fontsize=20)
    for i, dataset in enumerate(datasets):
        print(dataset)
        for j, model in enumerate(models):
            sub_df = df[(df["model"] == model) & (df["dataset"] == dataset)]
            # if len(sub_df) == 0:
            #     continue
            axes[i, j].set_title(f"{len(sub_df)} runs of {model} on {dataset} ")

            sub_df = plot_fn(sub_df, axes[i, j], dataset, model)
            sub_df["dataset"] = dataset
            sub_df["model"] = model

            new_df = new_df.append(sub_df)

    return new_df


def plot_all_datasets(df, model, plot_fn=make_plot_sweep(legend=True), fig_size=None):
    """Plot all dataset results for a single model."""
    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(len(datasets), 1, figsize=fig_size)
    # fig.suptitle(metric, fontsize=20)

    for i, dataset in enumerate(datasets):
        print(dataset)

        sub_df = df[(df["model"] == model) & (df["dataset"] == dataset)]
        axes[i].set_title(f"{len(sub_df)} runs of {model[:15]} on {dataset[:10]} ")

        plot_fn(sub_df, axes[i], dataset, model)
