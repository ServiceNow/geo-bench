"""Tests for Timm Generator."""

import os
import pickle

import pytest
from ruamel.yaml import YAML

from ccb.torch_toolbox.model_generators.timm_generator import TIMMGenerator


@pytest.mark.parametrize("backbone", ["resnet18", "convnext_base", "vit_tiny_patch16_224"])
def test_generate_timm_models(backbone):
    path = os.path.abspath("tests/data/ccb-test-classification/brick_kiln_v1.0/task_specs.pkl")
    with open(path, "rb") as f:
        task_specs = pickle.load(f)

    yaml = YAML()
    with open(os.path.join("tests", "configs", "base_classification.yaml"), "r") as yamlfile:
        config = yaml.load(yamlfile)

    model_gen = TIMMGenerator()
    config["model"]["backbone"] = backbone

    model = model_gen.generate_model(task_specs=task_specs, config=config)
    assert model.config["model"]["backbone"] == backbone


@pytest.mark.parametrize("init_method", ["random", "clone_random_rgb_channel"])
@pytest.mark.parametrize("backbone", ["resnet18", "convnext_base", "vit_tiny_patch16_224"])
def test_new_channel_init(init_method, backbone):
    path = os.path.abspath("tests/data/ccb-test-classification/brick_kiln_v1.0/task_specs.pkl")
    with open(path, "rb") as f:
        task_specs = pickle.load(f)

    yaml = YAML()
    with open(os.path.join("tests", "configs", "base_classification.yaml"), "r") as yamlfile:
        config = yaml.load(yamlfile)

    config["model"]["backbone"] = backbone
    config["model"]["new_channel_init_method"] = init_method
    config["dataset"]["band_names"] = ("red", "green", "blue", "05")

    model_gen = TIMMGenerator()

    model = model_gen.generate_model(task_specs=task_specs, config=config)
    assert model.config["model"]["backbone"] == backbone
