"""MAE Model Generator."""

from functools import partial
from typing import Any, Dict

import pytest
import timm.models.vision_transformer
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed  # with version 0.3.2
from torch import Tensor, nn
from torchvision import transforms

from ccb import io
from ccb.io.task import TaskSpecifications, imagenet_task_specs, mnist_task_specs
from ccb.torch_toolbox import model
from ccb.torch_toolbox.tests.test_toolbox import train_job_on_task

# from timm.models.layers.patch_embed import PatchEmbed for newer versions


class MaeGenerator(model.ModelGenerator):
    """Masked Auto-Encoder Model Generator.

    Masked Auto-Encoder model.

    """

    def generate(self, task_specs: TaskSpecifications, hyperparams: dict):
        """Return a model instance from task specs and hyperparameters.

        Args:
            task_specs: object with task specs
            hyperparams: dictionary containing hyperparameters

        Returns:
            model instance from task_specs and hyperparameters
        """
        backbone = Mae(self.model_path, task_specs, hyperparams)
        head = model.head_generator(task_specs, hyperparams)
        loss = model.train_loss_generator(task_specs, hyperparams)
        train_metrics = model.train_metrics_generator(task_specs, hyperparams)
        eval_metrics = model.eval_metrics_generator(task_specs, hyperparams)
        return model.Model(backbone, head, loss, hyperparams, train_metrics, eval_metrics)

    def get_collate_fn(self, task_specs: TaskSpecifications, hparams: dict):
        """Define a collate function to batch input tensors.

        Args:
            task_specs: task specs to retrieve dataset
            hyperparams: model hyperparameters

        Returns:
            collate function
        """
        if task_specs.dataset_name.lower() == "mnist":
            return None  # will use torch's default collate function.
        elif task_specs.dataset_name.lower() == "imagenet":
            return None  # will use torch's default collate function.
        else:
            return model.collate_rgb

    def get_transform(self, task_specs, hyperparams):
        """Define data transformations specific to the models generated.

        Args:
            task_specs: task specs to retrieve dataset
            hyperparams: model hyperparameters
            train: train mode true or false
            scale: define image scale
            ratio: define image ratio range

        Returns:
            callable function that applies transformations on input data
        """
        # These transforms are only valid for MNIST
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])  # Resize to 224x224
        return transform


def model_generator(hparams: Dict[str, Any] = {}) -> MaeGenerator:
    """Generate Mae generator with a defined set of hparams.

    Args:
        hparams: hyperparameters

    Returns:
        mae model generator
    """
    model_generator = MaeGenerator(hparams=hparams)
    return model_generator


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling.

    Default parameters correspond to Vit-Base/16
    """

    def __init__(
        self,
        img_size: int = 224,
        num_classes: int = 1000,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        norm_layer: bool = None,
        in_chans: int = 3,
        global_pool: bool = False,
        drop_path_rate: float = 0.1,
    ) -> None:
        """Initialize new instance of Vision Transformer.

        Args:
            img_size: image size dimension
            num_classes: number of classes to predict
            patch_size: size of patch
            embed_dim: size dimension of embedding
            depth:
            num_heads:
            mlp_ratio:
            qkv_bias: whether or not to apply bias term
            norm_layers: whether or not to use normalization layers
            in_chans: number of input channels
            global_pool: whether or not to apply global pooling
            drop_path_rate:

        """
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        super(VisionTransformer, self).__init__(
            num_classes=num_classes,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            drop_path_rate=drop_path_rate,
        )

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim
        )

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x: Tensor) -> Tensor:
        """Forward input through model.

        Args:
            x: input

        Returns:
            feature representation
        """
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome  # should be (batch size, feature_dim, 768 for ViT-base/16/224)


def vit_base_patch16(**kwargs):
    """Build Base VT with patch size 16.

    Args:
        **kwargs: Arbitrary keyword arguments.
    """
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    """Build Large VT with patch size 16.

    Args:
        **kwargs: Arbitrary keyword arguments.
    """
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_patch14(**kwargs):
    """Build Huge VT with patch size 14.

    Args:
        **kwargs: Arbitrary keyword arguments.
    """
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


class Mae(model.BackBone):
    """Masked Auto Encoder model.

    The Masked Auto Encoder model ...
    """

    def __init__(self, model_path, task_specs: io.TaskSpecifications, hyperparams):
        """Initialize a new instance of Mae model.

        Args:
            model_path: path to model
            task_specs: task specs to retrieve dataset
            hyperparams: model hyperparameters

        """
        super().__init__(model_path, task_specs, hyperparams)

        h = hyperparams

        if hyperparams["model_name"] == "vit_base_patch16":
            self.vit = vit_base_patch16(
                num_classes=h["num_classes"],
                drop_path_rate=h["drop_path_rate"],
                global_pool=h["global_pool"],
                in_chans=len(task_specs.bands_info),
            )
        elif hyperparams["model_name"] == "vit_large_patch16":
            self.vit = vit_large_patch16(
                num_classes=h["num_classes"],
                drop_path_rate=h["drop_path_rate"],
                global_pool=h["global_pool"],
                in_chans=len(task_specs.bands_info),
            )
        elif hyperparams["model_name"] == "vit_huge_patch14":
            self.vit = vit_huge_patch14(
                num_classes=h["num_classes"],
                drop_path_rate=h["drop_path_rate"],
                global_pool=h["global_pool"],
                in_chans=len(task_specs.bands_info),
            )

        self.vit = VisionTransformer(
            num_classes=1000,  # doesn't matter, we only want the backbone
            drop_path_rate=0.1,
            global_pool=True,
            in_chans=len(task_specs.bands_info),
        )

    def forward(self, x):
        """Forward input through model.

        Args:
            x: input

        Returns:
            feature representation
        """
        x = self.vit.forward_features(x)
        return x
