"""SSL MOCO pretrained model from ZhuLab."""

import copy
import random
from typing import Any, Callable, Dict

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data.dataloader import default_collate
from torchvision import models
from torchvision import transforms as tt

from ccb import io
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.model import (
    Model,
    ModelGenerator,
    eval_metrics_generator,
    head_generator,
    test_metrics_generator,
    train_loss_generator,
    train_metrics_generator,
)


class RSPretrained(ModelGenerator):
    """Remote Sensing Pretrained Checkpoints.

    Pretrained models on RGB data pretrained taken from:
    `An Empirical Study of Remote Sensing Pretraining <https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing>`_.
    """

    def __init__(self) -> None:
        """Initialize a new instance of RS Pretrained model."""
        super().__init__()

    def generate_model(self, task_specs: TaskSpecifications, config: dict) -> Model:
        """Return a ccb.torch_toolbox.model.Model instance from task specs and hparams.

        Args:
            task_specs: object with task specs
            hparams: dictionary containing hparams
            config: dictionary containing config

        Returns:
            configured model
        """
        if "resnet50" in config["model"]["backbone"]:
            backbone = models.resnet50(weights=None)
            backbone.fc = torch.nn.Linear(2048, 51)
            shapes = [(2048, 1, 1)]
            ckpt_path = "/mnt/data/experiments/nils/rs_pretrained_chkpts/rsp-resnet-50-ckpt.pth"

        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        previous_backbone = copy.deepcopy(backbone)
        backbone.load_state_dict(checkpoint["model"])

        prev_sample_weight = previous_backbone.layer4[0].conv1.weight

        post_sample_weight = backbone.layer4[0].conv1.weight

        # make sure there are new weights loaded
        assert torch.equal(prev_sample_weight, post_sample_weight) is False

        # replace head for the task at hand
        backbone.fc = torch.nn.Identity()
        head = head_generator(task_specs, shapes, config)
        loss = train_loss_generator(task_specs, config)
        train_metrics = train_metrics_generator(task_specs, config)
        eval_metrics = eval_metrics_generator(task_specs, config)
        test_metrics = test_metrics_generator(task_specs, config)

        return Model(
            backbone=backbone,
            head=head,
            loss_function=loss,
            config=config,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            test_metrics=test_metrics,
        )

    def get_collate_fn(self, task_specs: TaskSpecifications, config: dict):
        """Define a collate function to batch input tensors.

        Args:
            task_specs: task specs to retrieve dataset
            config: model hyperparameters

        Returns:
            collate function
        """
        return default_collate

    def get_transform(self, task_specs, config: Dict[str, Any], train=True) -> Callable[[io.Sample], Dict[str, Any]]:
        """Define data transformations specific to the models generated.

        Args:
            task_specs: task specs to retrieve dataset
            config: config file for dataset specifics
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
        if train:
            t.append(A.RandomRotate90(0.5))
            t.append(A.HorizontalFlip(0.5))
            t.append(A.VerticalFlip(0.5))
            t.append(A.Transpose(0.5))

        t.append(A.Resize(224, 224))

        # max_pixel_value = 1 is essential for us
        t.append(A.Normalize(mean=mean, std=std, max_pixel_value=1))
        t.append(ToTensorV2())
        transform_comp = A.Compose(t)

        def transform(sample: io.Sample):
            x: "np.typing.NDArray[np.float_]" = sample.pack_to_3d(band_names=config["dataset"]["band_names"])[0].astype(
                "float32"
            )
            x = transform_comp(image=x)["image"]
            return {"input": x, "label": sample.label}

        return transform


def model_generator() -> RSPretrained:
    """Return RSPretrained model generator.

    Returns:
        RSPretrained model generator
    """
    return RSPretrained()
