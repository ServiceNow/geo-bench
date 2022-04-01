from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
import pytest
from torchvision import transforms

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed  # with version 0.3.2

# from timm.models.layers.patch_embed import PatchEmbed for newer versions

from ccb import io
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox import model
from ccb.torch_toolbox.tests.test_toolbox import train_job_on_task
from ccb.io.task import mnist_task_specs, imagenet_task_specs
from ccb.experiment.experiment import hparams_to_string


class MaeGenerator(model.ModelGenerator):
    def generate(self, task_specs: TaskSpecifications, hyperparameters: dict):
        backbone = Mae(self.model_path, task_specs, hyperparameters)
        head = model.head_generator(task_specs, hyperparameters)
        loss = model.train_loss_generator(task_specs, hyperparameters)
        train_metrics = model.train_metrics_generator(task_specs, hyperparameters)
        eval_metrics = model.eval_metrics_generator(task_specs, hyperparameters)
        return model.Model(backbone, head, loss, hyperparameters, train_metrics, eval_metrics)

    def get_collate_fn(self, task_specs: TaskSpecifications, hparams: dict):
        if task_specs.dataset_name.lower() == "mnist":
            return None  # will use torch's default collate function.
        elif task_specs.dataset_name.lower() == "imagenet":
            return None  # will use torch's default collate function.
        else:
            return model.collate_rgb

    def hp_search(self, task_specs, max_num_configs=10):
        hparams = {
            "lr_milestones": (10, 20),
            "lr_gamma": 0.1,
            "lr_backbone": 1e-3,  # adjust for MAE
            "lr_head": 2e-3,
            "head_type": "linear",
            "train_iters": 50000,
            "features_shape": (768,),  # output dim of backbone, used by head_generator
            "loss_type": "crossentropy",
            "batch_size": 32,
            "num_workers": 4,
            #"logger": "csv",
            "logger": "wandb",
            "max_epochs": 1,
            "val_check_interval": 50,
            "limit_val_batches": 50,
            "limit_test_batches": 50,
            # Vit specific
            "model_name": "vit_base_patch16",
            "num_classes": 1000,  # doesn't matter, we only want the backbone
            "drop_path_rate": 0.1,
            "global_pool": True,
        }
        return hparams_to_string([hparams])

    def get_transform(self, task_specs, hyperparams):
        # These transforms are only valid for MNIST
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])  # Resize to 224x224
        return transform


model_generator = MaeGenerator()


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling
    Default parameters correspond to Vit-Base/16
    """

    def __init__(
        self,
        img_size=224,
        num_classes=1000,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=None,
        in_chans=3,
        global_pool=False,
        drop_path_rate=0.1,
    ):

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

    def forward_features(self, x):
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
    def __init__(self, model_path, task_specs: io.TaskSpecifications, hyperparams):
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
        x = self.vit.forward_features(x)
        return x


@pytest.mark.slow
def test_mae_mnist():
    train_job_on_task(model_generator, mnist_task_specs, 0.10)


@pytest.mark.slow
def test_mae_imagenet():
    train_job_on_task(model_generator, imagenet_task_specs, 0.10)


if __name__ == "__main__":
    # Using task specs, get imagenet
    test_mae_imagenet()
