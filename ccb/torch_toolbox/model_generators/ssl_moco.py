"""SSL MOCO pretrained model from ZhuLab."""

import copy
import random
from typing import Any, Callable, Dict

import albumentations as A
import numpy as np
import timm
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


def load_resnet_model(config):
    """Load resnet weights according to their code.

    Args:
        config: configuration file

    Returns:
        backbone
    """
    # this part comes from model loading from their script
    if "resnet50" in config["model"]["backbone"]:
        backbone = models.resnet50(weights=None)
        backbone.fc = torch.nn.Linear(2048, 19)
        ckpt_path = "/mnt/data/experiments/nils/ssl_checkpoints_zhu/B3_rn50_moco_0099_ckpt.pth"
        if len(config["dataset"]["band_names"]) >= 10:
            backbone.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            path_dict = {
                "moco": "/mnt/data/experiments/nils/ssl_checkpoints_zhu/B13_rn50_moco_0099_ckpt.pth",
                "dino": "/mnt/data/experiments/nils/ssl_checkpoints_zhu/B13_rn50_dino_0099_ckpt.pth",
            }
            ckpt_path = path_dict[config["model"]["ssl_method"]]

    elif "resnet18" in config["model"]["backbone"]:
        backbone = models.resnet18(weights=None)
        backbone.fc = torch.nn.Linear(512, 19)
        ckpt_path = "/mnt/data/experiments/nils/ssl_checkpoints_zhu/B3_rn18_moco_0199_ckpt.pth"

    print("=> loading checkpoint '{}'".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if config["model"]["ssl_method"] == "moco":
        state_dict = load_resnet_moco_state_dict(ckpt)
    elif config["model"]["ssl_method"] == "dino":
        state_dict = load_dino_state_dict(ckpt)

    previous_backbone = copy.deepcopy(backbone)
    prev_sample_weight = previous_backbone.layer4[0].conv1.weight

    msg = backbone.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    # backbone.load_state_dict(state_dict, strict=False)

    post_sample_weight = backbone.layer4[0].conv1.weight

    # make sure there are new weights loaded
    assert torch.equal(prev_sample_weight, post_sample_weight) is False

    before_check = backbone.conv1.weight[:, 0, :, :]

    backbone.conv1 = copy_sentinel_weights(backbone.conv1, config)
    after_check = backbone.conv1.weight[:, 0, :, :]

    assert torch.equal(before_check, after_check)

    return backbone


def copy_sentinel_weights(current_layer, config):
    """Copy weights."""
    new_layer = new_layer = torch.nn.Conv2d(
        in_channels=len(config["dataset"]["band_names"]),
        out_channels=current_layer.out_channels,
        kernel_size=current_layer.kernel_size,
        stride=current_layer.stride,
        padding=current_layer.padding,
    )
    # remove the 11th layer because we only have 12 bands
    new_indices = list(range(len(config["dataset"]["band_names"])))
    current_indices = list(range(13))
    if len(new_indices) == 12:
        current_indices.remove(10)
    elif len(new_indices) == 10:  # so2sat
        current_indices = [e for e in current_indices if e not in (3, 9, 10)]

    for new_idx, old_idx in zip(new_indices, current_indices):
        with torch.no_grad():
            new_layer.weight[:, new_idx : new_idx + 1, :, :] = current_layer.weight[:, old_idx : old_idx + 1, :, :]

    return new_layer


def load_resnet_moco_state_dict(ckpt):
    """Load Moco resnet weights."""
    state_dict = ckpt["state_dict"]

    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
            # remove prefix
            state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    return state_dict


def load_dino_state_dict(ckpt):
    """Load Dino weights for resnet and vit."""
    state_dict = ckpt["teacher"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    return state_dict


def load_vit_moco_state_dict(ckpt):
    """Load Moco weights for vit."""
    state_dict = ckpt["state_dict"]
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith("module.base_encoder") and not k.startswith("module.base_encoder.%s" % "head"):
            # remove prefix
            state_dict[k[len("module.base_encoder.") :]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    return state_dict


def load_vit_model(config):
    """Load Vit model weights."""
    backbone = timm.create_model(
        config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        features_only=False,
        in_chans=13,  # pretrained weights from zhu lab are for 13 bands
    )

    path_dict = {
        "moco": "/mnt/data/experiments/nils/ssl_checkpoints_zhu/B13_vits16_moco_0099_ckpt.pth",
        "dino": "/mnt/data/experiments/nils/ssl_checkpoints_zhu/B13_vits16_dino_0099_ckpt.pth",
        "data2vec": "/mnt/data/experiments/nils/ssl_checkpoints_zhu/B13_vits16_data2vec_0099_ckpt.pth",
    }
    ckpt_path = path_dict[config["model"]["ssl_method"]]

    print("=> loading checkpoint '{}'".format(ckpt_path))

    # moco
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # rename moco pre-trained keys
    if config["model"]["ssl_method"] == "moco":
        state_dict = load_vit_moco_state_dict(ckpt)
    elif config["model"]["ssl_method"] == "dino":
        state_dict = load_dino_state_dict(ckpt)

    msg = backbone.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"%s.weight" % "head", "%s.bias" % "head"}

    before_check = backbone.patch_embed.proj.weight[:, 1, :, :]

    backbone.patch_embed.proj = copy_sentinel_weights(backbone.patch_embed.proj, config)

    after_check = backbone.patch_embed.proj.weight[:, 1, :, :]

    assert torch.equal(before_check, after_check)

    return backbone


class SSLMocoGenerator(ModelGenerator):
    """SSL Moco Generator.

    SSL Moco models on RGB data pretrained taken from:
    `Zhu Lab <https://github.com/zhu-xlab/SSL4EO-S12>`_.
    """

    def __init__(self) -> None:
        """Initialize a new instance of SSL MOCO model."""
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
        if "resnet" in config["model"]["backbone"]:
            backbone = load_resnet_model(config)
            shapes = [(backbone.fc.in_features, 1, 1)]
            backbone.fc = torch.nn.Identity()
        elif "vit_small" in config["model"]["backbone"] and len(config["dataset"]["band_names"]) > 3:
            backbone = load_vit_model(config)
            print(backbone.patch_embed.proj)
            shapes = [(backbone.head.in_features, 1, 1)]
            backbone.head = torch.nn.Identity()

        # replace head for the task at hand
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


def model_generator() -> SSLMocoGenerator:
    """Return SSL Moco Generator.

    Returns:
        SSL Moco model generator
    """
    return SSLMocoGenerator()
