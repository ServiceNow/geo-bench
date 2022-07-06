import os
import pickle

import pytest
import torch
from ruamel.yaml import YAML

from ccb.torch_toolbox.model_generators.conv4 import Conv4, Conv4Generator


class TestConv4:
    def test_in_channels(self) -> None:
        path = os.path.abspath("tests/data/ccb-test-classification/brick_kiln_v1.0/task_specs.pkl")
        with open(path, "rb") as f:
            task_specs = pickle.load(f)

        hparams = {
            "logger": "csv",
            "lr_backbone": 1e-3,
            "lr_head": 1e-2,
            "optimizer": "sgd",
            "momentum": 0.9,
            "batch_size": 32,
            "max_epochs": 1,
            "band_names": ["red", "green", "blue"],
            "num_workers": 0,
            "seed": 1,
            "format": "hdf5",
        }
        model = Conv4(model_path=".", task_specs=task_specs, hparams=hparams)
        x = torch.randn(2, 3, 64, 64)
        model(x)

        # manipulate task_specs.bands_info for the length of bands
        setattr(task_specs, "bands_info", list(range(0, 1)))
        wrong_model = Conv4(model_path=".", task_specs=task_specs, hparams=hparams)
        match = "to have 1 channels, but got 3 channels instead"
        with pytest.raises(RuntimeError, match=match):
            wrong_model(x)


def test_generate_conv4_models():
    path = os.path.abspath("tests/data/ccb-test-classification/brick_kiln_v1.0/task_specs.pkl")
    with open(path, "rb") as f:
        task_specs = pickle.load(f)

    yaml = YAML()
    with open(os.path.join("tests", "configs", "base_classification.yaml"), "r") as yamlfile:
        config = yaml.load(yamlfile)

    with open(os.path.join("tests", "configs", "classification_hparams.yaml"), "r") as yamlfile:
        hparams = yaml.load(yamlfile)

    hparams["backbone"] = "conv4"
    model_gen = Conv4Generator()
    model = model_gen.generate_model(task_specs=task_specs, hparams=hparams, config=config)
    assert model.hyperparameters["backbone"] == "conv4"
