"""Conv4 Model Generator."""

from typing import Any, Callable, Dict, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as tt

from ccb import io
from ccb.io.dataset import Band
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.model import (
    BackBone,
    Model,
    ModelGenerator,
    eval_metrics_generator,
    head_generator,
    test_metrics_generator,
    train_loss_generator,
    train_metrics_generator,
)


class Conv4Generator(ModelGenerator):
    """Conv4Generator.

    Model generator for a simple 4 layer convolutional neural network.
    """

    def __init__(
        self,
    ) -> None:
        """Initialize a new instance of Conv4 model generator."""
        super().__init__()

    def generate_model(self, task_specs: TaskSpecifications, config: Dict[str, Any]) -> Model:
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
        test_metrics = test_metrics_generator(task_specs, config)
        return Model(backbone, head, loss, config, train_metrics, eval_metrics, test_metrics)

    def get_collate_fn(self, task_specs: TaskSpecifications, config: Dict[str, Any]) -> Any:
        """Define a collate function to batch input tensors.

        Args:
            task_specs: task specs to retrieve dataset
            config: config

        Returns:
            collate function
        """
        return default_collate

    def get_transform(
        self,
        task_specs: TaskSpecifications,
        config: Dict[str, Any],
        train: bool = True,
    ) -> Callable[[io.Sample], Dict[str, Any]]:
        """Define data transformations specific to the models generated.

        Args:
            task_specs: task specs to retrieve dataset
            config: config file
            train: train mode true or false

        Returns:
            callable function that applies transformations on input data
        """
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

        t.append(tt.Resize(224))
        transform_comp = tt.Compose(t)

        def transform(
            sample: io.Sample,
        ) -> Dict[str, Union["np.typing.NDArray[np.float_]", Band, int]]:
            x = sample.pack_to_3d(band_names=(config["dataset"]["band_names"]))[0].astype("float32")
            x = transform_comp(x)
            return {"input": x, "label": sample.label}

        return transform


def model_generator() -> Conv4Generator:
    """Generate Conv generator.

    Returns:
        conv model generator
    """
    return Conv4Generator()


class Conv4(BackBone):
    """Conv4 model.

    Simple convolutional neural net with 4 layers.
    """

    def __init__(self, model_path: str, task_specs: io.TaskSpecifications, config: Dict[str, Any]) -> None:
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
