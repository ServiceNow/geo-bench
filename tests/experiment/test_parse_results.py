from ccb.experiment.parse_results import ExpResult
from pathlib import Path


def test_exp_result():
    log_dir = Path(__file__).parent.parent / "data" / "log_dir" / "csv_logs" / "version_0"
    exp_result = ExpResult(log_dir)

    traces = exp_result.get_traces()
    best_point = exp_result.get_best_point(filt_size=1)

    best_step = best_point["best_step"]
    val_score = traces["val_metric"][best_step]

    assert traces["val_metric"].max() == val_score
    assert val_score == best_point["val_metric"]

    task = exp_result.get_task_specs()
    assert task.dataset_name == "bigearthnet"

    hparams = exp_result.get_hparams()
    assert hparams["batch_size"] == 16

    exp_info = exp_result.get_combined_info()
    hparams["batch_size"] = exp_info["batch_size"]
    for key, val in exp_info.items():
        print(f"{key}: {str(val)[:100]}")


if __name__ == "__main__":
    test_exp_result()
