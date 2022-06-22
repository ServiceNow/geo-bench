"""Tests for Timm Generator."""

import torch
import pickle
import os
import pytest
from ccb.torch_toolbox.model_generators.timm_generator import TIMMGenerator



@pytest.mark.parametrize("backbone", ["resnet18", "convnext_base", "vit_tiny_patch16_224"])
def test_generate_models(backbone):
    path = os.path.abspath("tests/data/ccb-test-classification/brick_kiln_v1.0/task_specs.pkl")
    with open(path, "rb") as f:
        task_specs = pickle.load(f)

    hparams = {
        "backbone": backbone,
        "pretrained": True,
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

    model_generator = TIMMGenerator(hparams=hparams)

    model = model_generator.generate(task_specs, model_generator.base_hparams)

@pytest.mark.parametrize("init_method", ["random", "clone_random_rgb_channel"])
def test_new_channel_init(init_method):
    path = os.path.abspath("tests/data/ccb-test-classification/brick_kiln_v1.0/task_specs.pkl")
    with open(path, "rb") as f:
        task_specs = pickle.load(f)

    hparams = {
        "backbone": "resnet18",
        "pretrained": True,
        "logger": "csv",
        "lr_backbone": 1e-3,
        "lr_head": 1e-2,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
        "band_names": ["red", "green", "blue", "05"],
        "num_workers": 0,
        "seed": 1,
        "format": "hdf5",
        "new_channel_init_method": init_method
    }

    model_generator = TIMMGenerator(hparams=hparams)

    model = model_generator.generate(task_specs, model_generator.base_hparams)
