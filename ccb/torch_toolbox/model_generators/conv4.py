"""Conv4 Model Generator."""

from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as tt

from ccb import io
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.model import (
    BackBone,
    Model,
    ModelGenerator,
    eval_metrics_generator,
    head_generator,
    train_loss_generator,
    train_metrics_generator,
)


class Conv4Generator(ModelGenerator):
    """Conv4Generator.

    Model generator for a simple 4 layer convolutional neural network.
    """

    def __init__(self, config=None) -> None:
        """Initialize a new instance of Conv4 model generator.

        Args:
            hparams: set of hyperparameters
        """
        super().__init__()

        # self.base_hparams = {
        #     "backbone": "conv4",
        #     "lr_gamma": 0.1,
        #     "lr_backbone": 4e-3,
        #     "lr_head": 4e-3,
        #     "head_type": "linear",
        #     "hidden_size": 128,
        #     "loss_type": "crossentropy",
        #     "batch_size": 32,
        #     "num_workers": 0,
        #     "max_epochs": 1,
        #     "n_gpus": 0,
        #     "logger": "csv",
        #     "sweep_config_yaml_path": "/mnt/home/climate-change-benchmark/ccb/torch_toolbox/wandb/hparams_classification_conv4.yaml",
        #     "num_seeds": 3,
        #     "num_agents": 4,
        #     "num_trials_per_agent": 5,
        #     "band_names": ["red", "green", "blue"],
        #     "image_size": 224,
        #     "format": "hdf5",
        # }
        # if hparams is not None:
        #     self.base_hparams.update(hparams)

    def generate_model(self, task_specs: TaskSpecifications, config: dict) -> Model:
        """Return a model instance from task specs and hyperparameters.

        Args:
            task_specs: object with task specs
            hparams: dictionary containing hyperparameters

        Returns:
            model instance from task_specs and hyperparameters
        """
        backbone = Conv4(self.model_path, task_specs, config)
        head = head_generator(task_specs, [(64,)], config)
        loss = train_loss_generator(task_specs, config)
        train_metrics = train_metrics_generator(task_specs, config)
        eval_metrics = eval_metrics_generator(task_specs, config)
        return Model(backbone, head, loss, config, train_metrics, eval_metrics)

    def get_collate_fn(self, task_specs: TaskSpecifications, config: dict):
        """Define a collate function to batch input tensors.

        Args:
            task_specs: task specs to retrieve dataset
            config: config

        Returns:
            collate function
        """
        return default_collate

    def get_transform(self, task_specs, config: Dict[str, Any], train=True, scale=None, ratio=None):
        """Define data transformations specific to the models generated.

        Args:
            task_specs: task specs to retrieve dataset
            config: config file
            train: train mode true or false
            scale: define image scale
            ratio: define image ratio range

        Returns:
            callable function that applies transformations on input data
        """
        scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
        ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
        _, h, w = (len(config["dataset"]["band_names"]), config["model"]["image_size"], config["model"]["image_size"])

        mean, std = task_specs.get_dataset(
            split="train",
            format=config["dataset"]["format"],
            band_names=tuple(config["dataset"]["band_names"]),
            benchmark_dir=config["experiment"]["benchmark_dir"],
            partition_name=config["experiment"]["partition_name"],
        ).normalization_stats()
        t = []
        t.append(tt.ToTensor())
        t.append(tt.Normalize(mean=mean, std=std))
        if train:
            t.append(tt.RandomHorizontalFlip())
            t.append(tt.RandomResizedCrop((h, w), scale=scale, ratio=ratio))

        t.append(tt.Resize((config["model"]["image_size"], config["model"]["image_size"])))

        transform_comp = tt.Compose(t)

        def transform(sample: io.Sample):
            x = sample.pack_to_3d(band_names=(config["dataset"]["band_names"]))[0].astype("float32")
            x = transform_comp(x)
            return {"input": x, "label": sample.label}

        return transform


def model_generator(config: Dict[str, Any] = {}) -> Conv4Generator:
    """Generate Conv generator with a defined set of hparams.

    Args:
        config: config

    Returns:
        conv model generator
    """
    model_generator = Conv4Generator(config=config)
    return model_generator


class Conv4(BackBone):
    """Conv4 model.

    Simple convolutional neural net with 4 layers.
    """

    def __init__(self, model_path: str, task_specs: io.TaskSpecifications, config) -> None:
        """Initialize a new instance of Conv4 model.

        Args:
            model_path: path to model
            task_specs: task specs to retrieve dataset
            hparams: model hyperparameters

        """
        super().__init__(model_path, task_specs, config)
        if task_specs.bands_info is not None:
            n_bands = min(3, len(task_specs.bands_info))
        self.conv0 = torch.nn.Conv2d(n_bands, 64, 3, 1, 1)
        self.conv1 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward input through model.

        Args:
            x: input

        Returns:
            feature representation
        """
        x = F.relu(self.conv0(x), True)
        x = F.relu(self.conv1(x), True)
        x = F.max_pool2d(x, 3, 2, 1)
        x = F.relu(self.conv2(x), True)
        x = F.max_pool2d(x, 3, 2, 1)
        x = F.relu(self.conv3(x), True)
        x = F.max_pool2d(x, 3, 2, 1)
        return x.mean((2, 3))
