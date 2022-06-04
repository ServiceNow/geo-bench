from typing import List, Dict, Any
from ccb import io
from ccb.experiment.experiment import hparams_to_string
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.model import (
    BackBone,
    ModelGenerator,
    Model,
    train_loss_generator,
    train_metrics_generator,
    eval_metrics_generator,
    head_generator,
    collate_rgb,
)
from torch.utils.data.dataloader import default_collate
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms as tt
import logging
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF


class SegmentationGenerator(ModelGenerator):
    """SegmentationGenerator
    This ModelGenerator uses segmentation_models.pytorch as backbone
    See its documentation: https://github.com/qubvel/segmentation_models.pytorch
    It supports TIMM encoders! https://smp.readthedocs.io/en/latest/encoders_timm.html
    """

    def __init__(self, hparams=None) -> None:
        super().__init__()
        # These params are for unit tests, please set proper ones for real optimization
        self.base_hparams = {
            "input_size": (3, 64, 64),  # FIXME
            "pretrained": True,
            "lr_backbone": 1e-5,
            "lr_head": 1e-4,
            "optimizer": "sgd",
            "head_type": "linear",
            "loss_type": "crossentropy",
            "batch_size": 4,
            "num_workers": 1,
            "max_epochs": 10,
            "n_gpus": 1,
            "logger": "csv",  # Set to wandb for wandb tracking
            "encoder_type": "resnet18",
            "decoder_type": "Unet",
            "decoder_weights": "imagenet",
            "enable_progress_bar": False,
            "log_segmentation_masks": False,  # Set to true for visualizing seg masks in wandb
            "fast_dev_run": True,  # runs 1 train, 1 validation, and 1 test batch.
            "format": "hdf5",
        }
        if hparams is not None:
            self.base_hparams.update(hparams)

    def generate(self, task_specs: TaskSpecifications, hyperparameters: dict):
        """Returns a ccb.torch_toolbox.model.Model instance from task specs
           and hyperparameters

        Args:
            task_specs (TaskSpecifications): object with task specs
            hyperparameters (dict): dictionary containing hyperparameters
        """
        encoder_type = hyperparameters["encoder_type"]
        decoder_type = hyperparameters["decoder_type"]
        encoder_weights = hyperparameters.get("encoder_weights", None)
        # in_ch, *other_dims = features_shape[-1]
        out_ch = task_specs.label_type.n_classes

        # Load segmentation backbone from py-segmentation-models
        backbone = getattr(smp, decoder_type)(
            encoder_name=encoder_type,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=hyperparameters["input_size"][
                0
            ],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=out_ch,
        )  # model output channels (number of classes in your dataset))

        # For timm models, we can extract the mean and std of the pre-trained backbone
        # hyperparameters.update({"mean": backbone.default_cfg["mean"]})
        # hyperparameters.update({"std": backbone.default_cfg["std"]})

        with torch.no_grad():
            backbone.eval()
            features = torch.zeros(hyperparameters["input_size"]).unsqueeze(0)
            features = backbone.encoder(features)

        class Noop(torch.nn.Module):
            def forward(self, x):
                return x

        head = Noop()  # pytorch image models already adds a classifier on top of the UNETs
        # head = head_generator(task_specs, shapes, hyperparameters)
        loss = train_loss_generator(task_specs, hyperparameters)
        train_metrics = train_metrics_generator(task_specs, hyperparameters)
        eval_metrics = eval_metrics_generator(task_specs, hyperparameters)
        return Model(backbone, head, loss, hyperparameters, train_metrics, eval_metrics)

    def hp_search(self, task_specs, max_num_configs=10):

        hparams2 = self.base_hparams.copy()
        hparams2["lr_head"] = 4e-3

        return hparams_to_string([self.base_hparams, hparams2])

    def get_collate_fn(self, task_specs: TaskSpecifications, hparams: dict):
        return default_collate

    def get_transform(self, task_specs, hyperparams, train=True, scale=None, ratio=None):
        scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
        ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
        c, h, w = hyperparams["input_size"]
        mean, std = task_specs.get_dataset(split="train").rgb_stats()

        class SegTransform:
            """
            This is a helper class that helps applying the same transformation
            to input images and segmentation masks.
            """

            def __call__(self, x, resample=True, train=True):
                """Applies data augmentation to input and segmentation mask

                Args:
                    x (torch.Tensor): input image or segmentation mask
                    resample (bool, optional): whether to resample (True) or reuse (False) previous transforms.
                                               Defaults to True.
                    train (bool, optional): whether in training mode. No aug during validation. Defaults to True.

                Returns:
                    _type_: _description_
                """
                if train:
                    if resample:
                        self.crop_params = tt.RandomResizedCrop.get_params(
                            x, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)
                        )
                        self.flip = bool(torch.randint(0, 2, size=(1,)))
                    x = TF.resized_crop(x, *self.crop_params, size=(h, w))
                    if self.flip:
                        x = TF.hflip(x)
                else:
                    x = TF.resize(x, (h, w))
                return x

        def transform(sample: io.Sample):
            t_x = []
            st = SegTransform()
            t_x.append(tt.ToTensor())
            t_x.append(tt.Normalize(mean=mean, std=std))
            t_x.append(lambda x: st(x, resample=True, train=train))
            t_x = tt.Compose(t_x)
            t_y = []
            t_y.append(tt.ToTensor())
            t_y.append(lambda x: st(x, resample=False, train=train))
            t_y = tt.Compose(t_y)

            x = sample.pack_to_3d(band_names=("red", "green", "blue"))[0].astype("float32")
            x, y = t_x(x), t_y(sample.label.data.astype("float32"))
            return {"input": x, "label": y.long().squeeze()}

        return transform


def model_generator(hparams: Dict[str, Any] = {}) -> SegmentationGenerator:
    model_generator = SegmentationGenerator(hparams=hparams)
    return model_generator
