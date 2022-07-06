"""Test model.py"""

import copy
import os
import pickle

import pytest
import torch
from ruamel.yaml import YAML

from ccb import io
from ccb.torch_toolbox.model import Model, ModelGenerator, _balanced_binary_cross_entropy_with_logits, head_generator
from ccb.torch_toolbox.model_generators.timm_generator import TIMMGenerator
from ccb.torch_toolbox.modules import ClassificationHead


class TestModel:
    @pytest.fixture
    def model(self):
        path = os.path.abspath("tests/data/ccb-test-classification/brick_kiln_v1.0/task_specs.pkl")
        with open(path, "rb") as f:
            task_specs = pickle.load(f)

        yaml = YAML()
        with open(os.path.join("tests", "configs", "base_classification.yaml"), "r") as yamlfile:
            config = yaml.load(yamlfile)

        with open(os.path.join("tests", "configs", "classification_hparams.yaml"), "r") as yamlfile:
            hparams = yaml.load(yamlfile)

        model_gen = TIMMGenerator()
        hparams["backbone"] = "resnet18"

        return model_gen.generate_model(task_specs=task_specs, hparams=hparams, config=config)

    def test_forward(self, model: Model) -> None:
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape[-1] == model.head.num_classes

    def test_forward_frozen_backbone(self, model: Model) -> None:
        model.hyperparameters["lr_backbone"] = 0
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape[-1] == model.head.num_classes

    def test_optimizers(self, model: Model) -> None:
        for opt, instc in zip(["sgd", "adam", "adamw"], [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW]):
            model.hyperparameters["optimizer"] = opt
            optims = model.configure_optimizers()
            assert isinstance(optims[0], instc)

    def test_scheduler(self, model: Model) -> None:
        model.hyperparameters["scheduler"] = "step"
        model.hyperparameters["lr_milestones"] = [1, 2]
        model.hyperparameters["lr_gamma"] = 0.1
        _, scheduler = model.configure_optimizers()
        assert isinstance(scheduler[0], torch.optim.lr_scheduler.MultiStepLR)


class TestGenerator:
    path = os.path.abspath("tests/data/ccb-test-classification/brick_kiln_v1.0/task_specs.pkl")
    with open(path, "rb") as f:
        task_specs = pickle.load(f)

    yaml = YAML()
    with open(os.path.join("tests", "configs", "classification_hparams.yaml"), "r") as yamlfile:
        hparams = yaml.load(yamlfile)

    @pytest.fixture
    def model_gen(self):
        class TestModelGenerator(ModelGenerator):
            def __init__(self):
                super().__init__()

        return TestModelGenerator()

    def test_generate_model_present(self, model_gen: ModelGenerator) -> None:
        with pytest.raises(NotImplementedError, match="Necessary to"):
            model_gen.generate_model(task_specs=self.task_specs, hparams=self.hparams)

    def test_get_collate_fn_present(self, model_gen: ModelGenerator) -> None:
        with pytest.raises(NotImplementedError, match="Necessary to"):
            model_gen.get_collate_fn(task_specs=self.task_specs, hparams=self.hparams)

    def test_get_transform_present(self, model_gen: ModelGenerator) -> None:
        with pytest.raises(NotImplementedError, match="Necessary to"):
            model_gen.get_transform(task_specs=self.task_specs, hparams=self.hparams, train=True)


class TestHeadGenerator:
    path = os.path.abspath("tests/data/ccb-test-classification/brick_kiln_v1.0/task_specs.pkl")
    with open(path, "rb") as f:
        task_specs = pickle.load(f)

    yaml = YAML()
    with open(os.path.join("tests", "configs", "classification_hparams.yaml"), "r") as yamlfile:
        hparams = yaml.load(yamlfile)

    def test_valid_head_generator(self):
        hparams = self.hparams.copy()
        hparams["head_type"] = "linear"
        head = head_generator(task_specs=self.task_specs, features_shape=[(3, 10, 10)], hyperparams=hparams)
        assert isinstance(head, ClassificationHead)

    def test_invalid_head_generator(self):
        hparams = self.hparams.copy()
        hparams["head_type"] = "foo"
        with pytest.raises(AssertionError):
            head_generator(task_specs=self.task_specs, features_shape=[(3, 10, 10)], hyperparams=hparams)

    def test_multilabel_task(self):
        task_specs = copy.copy(self.task_specs)
        setattr(task_specs, "label_type", io.MultiLabelClassification(n_classes=2, class_names=["foo", "bar"]))
        head = head_generator(task_specs=task_specs, features_shape=[(3, 10, 10)], hyperparams=self.hparams)
        assert isinstance(head, ClassificationHead)

    def test_invalid_task(self):
        task_specs = copy.copy(self.task_specs)
        setattr(task_specs, "label_type", torch.nn.Module)
        with pytest.raises(ValueError, match="Unrecognized task"):
            head_generator(task_specs=task_specs, features_shape=[(3, 10, 10)], hyperparams=self.hparams)


def test_balanced_binary_cross_entropy():
    target = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    output = torch.tensor([[100.0, -100.0, 100.0], [-100.0, 100.0, -100.0]])
    loss = _balanced_binary_cross_entropy_with_logits(outputs=output, targets=target)
    assert loss == 0
