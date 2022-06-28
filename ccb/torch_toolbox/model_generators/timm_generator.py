"""Timm Model Generator."""

import os
import random
import string
from typing import Any, Callable, Dict

import pytorch_lightning as pl
import timm
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as tt

from ccb import io
from ccb.experiment.experiment import Job
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.model import (
    Model,
    ModelGenerator,
    eval_metrics_generator,
    head_generator,
    train_loss_generator,
    train_metrics_generator,
)


class TIMMGenerator(ModelGenerator):
    """Timm Model Generator.

    The Timm Model Generator lets you define a model with any available
    `Timm models <https://rwightman.github.io/pytorch-image-models/>`_ as a backbone, and
    attaches a classification head according to the task.
    """

    def __init__(self, hparams=None) -> None:
        """Initialize a new instance of Timm model generator.

        Args:
            hparams: set of hyperparameters
        """
        super().__init__()

        self.base_hparams = {
            "backbone": "resnet50",  # resnet18, convnext_base, vit_tiny_patch16_224, vit_small_patch16_224. swinv2_tiny_window16_256
            "pretrained": True,
            "lr_backbone": 1e-6,
            "lr_head": 1e-4,
            "optimizer": "sgd",
            "head_type": "linear",
            "hidden_size": 512,
            "loss_type": "crossentropy",
            "batch_size": 64,
            "num_workers": 0,
            "max_epochs": 1,
            "n_gpus": 0,
            "logger": "wandb",
            "sweep_config_yaml_path": "/mnt/home/climate-change-benchmark/ccb/torch_toolbox/wandb/hparams_classification_resnet50.yaml",
            "num_agents": 4,
            "num_trials_per_agent": 5,
            "band_names": ["red", "green", "blue"],  # , "01", "05", "06", "07", "08", "08A", "09", "10", "11", "12"],
            "image_size": 224,
            "format": "hdf5",
            "new_channel_init_method": "random",  # random, clone_random_rgb_channel
        }
        if hparams is not None:
            self.base_hparams.update(hparams)

    def generate_model(self, task_specs: TaskSpecifications, hparams: dict, config: dict) -> Model:
        """Return a ccb.torch_toolbox.model.Model instance from task specs and hparams.

        Args:
            task_specs: object with task specs
            hparams: dictionary containing hparams
            config: dictionary containing config

        Returns:
            configured model
        """
        backbone = timm.create_model(hparams["backbone"], pretrained=hparams["pretrained"], features_only=False)
        setattr(backbone, backbone.default_cfg["classifier"], torch.nn.Identity())

        new_in_channels = len(config["dataset"]["band_names"])
        # if we go beyond RGB channels need to initialize other layers, otherwise keep the same
        if hparams["backbone"] in ["resnet18", "resnet50"] and new_in_channels > 3:
            current_layer = backbone.conv1

            # Creating new Conv2d layer
            new_layer = torch.nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=current_layer.out_channels,
                kernel_size=current_layer.kernel_size,
                stride=current_layer.stride,
                padding=current_layer.padding,
                bias=current_layer.bias,
            )

            backbone.conv1 = self._initialize_additional_in_channels(
                current_layer=current_layer, new_layer=new_layer, task_specs=task_specs, config=config
            )

        elif hparams["backbone"] in ["convnext_base"] and new_in_channels > 3:
            current_layer = backbone.stem[0]

            # Creating new Conv2d layer
            new_layer = torch.nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=current_layer.out_channels,
                kernel_size=current_layer.kernel_size,
                stride=current_layer.stride,
                padding=current_layer.padding,
            )

            new_layer.bias.data = current_layer.bias

            # add new layer back to backbone
            backbone.stem[0] = self._initialize_additional_in_channels(
                current_layer=current_layer, new_layer=new_layer, task_specs=task_specs, config=config
            )

        elif (
            hparams["backbone"]
            in [
                "vit_tiny_patch16_224",
                "vit_small_patch16_224",
                "swinv2_tiny_window16_256",
            ]
            and new_in_channels > 3
        ):
            current_layer = backbone.patch_embed.proj

            # Creating new Conv2d layer
            new_layer = torch.nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=current_layer.out_channels,
                kernel_size=current_layer.kernel_size,
                stride=current_layer.stride,
                padding=current_layer.padding,
            )

            new_layer.bias.data = current_layer.bias

            backbone.patch_embed.proj = self._initialize_additional_in_channels(
                current_layer=current_layer, new_layer=new_layer, task_specs=task_specs, config=config
            )

        config["model"]["input_size"] = (
            len(config["dataset"]["band_names"]),
            hparams["image_size"],
            hparams["image_size"],
        )

        with torch.no_grad():
            backbone.eval()
            features = torch.zeros(config["model"]["input_size"]).unsqueeze(0)
            features = backbone(features)
        shapes = [features.shape[1:]]  # get the backbone's output features

        config["model"]["n_backbone_features"] = shapes[0][0]

        head = head_generator(task_specs, shapes, hparams)
        loss = train_loss_generator(task_specs, hparams)
        train_metrics = train_metrics_generator(task_specs, hparams)
        eval_metrics = eval_metrics_generator(task_specs, hparams)

        return Model(backbone, head, loss, hparams, train_metrics, eval_metrics)

    def _initialize_additional_in_channels(
        self,
        current_layer: torch.nn.Conv2d,
        new_layer: torch.nn.Conv2d,
        task_specs: TaskSpecifications,
        config: Dict[str, Any],
    ) -> torch.nn.Conv2d:
        """Initialize new additional input channels.

        By default RGB channels are copied and new input channels randomly initialized

        Args:
            current_layer: current Conv2d backbone layer
            new_layer: newly initialized layer to which to copy weights
            task_specs: task specs to retrieve dataset
            config: config file for dataset specifics


        Returns:
            newly initialized input Conv2d layer
        """
        method = config["model"]["new_channel_init_method"]

        dataset = task_specs.get_dataset(
            split="train",
            band_names=config["dataset"]["band_names"],
            format=config["dataset"]["format"],
            partition_name=config["experiment"]["partition_name"],
        )
        alt_band_names = dataset.alt_band_names
        band_names = dataset.band_names

        # find index of the rgb bands
        new_rgb_indices = []
        full_rgb_names = []
        for rgb_name in ["red", "green", "blue"]:
            rgb_full_name = alt_band_names[rgb_name]
            new_rgb_indices.append(band_names.index(rgb_full_name))
            full_rgb_names.append(rgb_full_name)

        non_rgb_names = list(set(band_names) - set(full_rgb_names))
        non_rgb_indices = [band_names.index(band) for band in non_rgb_names]

        # how rgb is ordered in current layer
        current_rgb_indices = [0, 1, 2]
        # need to check that this order matches with how the data of all bands is retrieved
        for new_idx, old_idx in zip(new_rgb_indices, current_rgb_indices):
            with torch.no_grad():
                new_layer.weight[:, new_idx : new_idx + 1, :, :] = current_layer.weight[:, old_idx : old_idx + 1, :, :]

        ## can define different approaches here about how to initialize other channels
        if method == "clone_random_rgb_channel":
            # Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
            for channel in non_rgb_indices:
                # index of existing channel weights
                # Here will initialize the weights in new channel by randomly cloning one pretrained channel and adding gaussian noise
                rand_rgb_idx = random.randint(0, 2)
                # find respective location of rgb pands in old and new
                current_rgb_idx = current_rgb_indices.index(rand_rgb_idx)

                with torch.no_grad():
                    new_layer.weight[:, channel : channel + 1, :, :] = current_layer.weight[
                        :, current_rgb_idx : current_rgb_idx + 1, ::
                    ].clone() + random.gauss(0, 1 / new_layer.in_channels)

                new_layer.weight = torch.nn.Parameter(new_layer.weight)

        return new_layer

    def get_collate_fn(self, task_specs: TaskSpecifications, hparams: dict):
        """Define a collate function to batch input tensors.

        Args:
            task_specs: task specs to retrieve dataset
            hparams: model hyperparameters

        Returns:
            collate function
        """
        return default_collate

    def get_transform(
        self, task_specs, config: Dict[str, Any], hparams: Dict[str, Any], train=True, scale=None, ratio=None
    ) -> Callable[[io.Sample], Dict[str, Any]]:
        """Define data transformations specific to the models generated.

        Args:
            task_specs: task specs to retrieve dataset
            config: config file for dataset specifics
            hparams: hparam file for model specifics
            train: train mode true or false
            scale: define image scale
            ratio: define image ratio range

        Returns:
            callable function that applies transformations on input data
        """
        scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
        ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
        _, h, w = config["model"]["input_size"]

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

        t.append(tt.Resize((hparams["image_size"], hparams["image_size"])))

        t = tt.Compose(t)

        def transform(sample: io.Sample):
            x = sample.pack_to_3d(band_names=tuple(config["dataset"]["band_names"]))[0].astype("float32")
            x = t(x)
            return {"input": x, "label": sample.label}

        return transform


def model_generator() -> TIMMGenerator:
    """Initializ Timm Generator.

    Returns:
        segmentation model generator
    """
    return TIMMGenerator()
