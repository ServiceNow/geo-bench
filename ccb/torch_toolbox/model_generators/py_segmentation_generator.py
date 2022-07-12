"""Segmentation Model Generator."""

from typing import Any, Dict

import segmentation_models_pytorch as smp
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as tt

from ccb import io
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.model import (
    Model,
    ModelGenerator,
    eval_metrics_generator,
    train_loss_generator,
    train_metrics_generator,
)


class SegmentationGenerator(ModelGenerator):
    """SegmentationGenerator.

    This ModelGenerator uses
    `segmentation_models.pytorch <https://github.com/qubvel/segmentation_models.pytorch>`_
    as an architecture choice and allows any of these
    `TIMM encoders <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
    """

    def __init__(self, hparams=None) -> None:
        """Initialize a new instance of segmentation generator.

        Args:
            hparams: set of hyperparameters

        """
        super().__init__()

        # These params are for unit tests, please set proper ones for real optimization
        # self.base_hparams = {
        #     "input_size": (3, 256, 256),  # FIXME
        #     "pretrained": True,
        #     "lr_backbone": 1e-5,
        #     "lr_head": 1e-4,
        #     "optimizer": "adamw",
        #     "head_type": "linear",
        #     "loss_type": "crossentropy",
        #     "batch_size": 1,
        #     "num_workers": 0,
        #     "max_epochs": 1,
        #     "n_gpus": 0,
        #     "logger": "csv",  # Set to wandb for wandb tracking
        #     "encoder_type": "resnet18",
        #     "accumulate_grad_batches": 2,
        #     "decoder_type": "Unet",
        #     "decoder_weights": "imagenet",
        #     "enable_progress_bar": False,
        #     "log_segmentation_masks": False,  # Set to true for visualizing seg masks in wandb
        #     "fast_dev_run": False,  # runs 1 train, 1 validation, and 1 test batch.
        #     "sweep_config_yaml_path": "/mnt/home/climate-change-benchmark/ccb/torch_toolbox/wandb/hparams_segmentation_resnet101_deeplabv3.yaml",
        #     "num_agents": 4,
        #     "num_trials_per_agent": 5,
        #     "band_names": ["red", "green", "blue"],  # , "01", "05", "06", "07", "08", "08A", "09", "10", "11", "12"],
        #     "image_size": 224,
        #     "format": "hdf5",
        # }
        # if hparams is not None:
        #     self.base_hparams.update(hparams)

    def generate_model(self, task_specs: TaskSpecifications, config: dict) -> Model:
        """Return model instance from task specs and hyperparameters.

        Args:
            task_specs: object with task specs
            config: config: dictionary containing config

        Returns:
            model specified by task specs and hyperparameters
        """
        encoder_type = config["model"]["encoder_type"]
        decoder_type = config["model"]["decoder_type"]
        encoder_weights = config["model"].get("encoder_weights", None)
        # in_ch, *other_dims = features_shape[-1]
        out_ch = task_specs.label_type.n_classes

        # Load segmentation backbone from py-segmentation-models
        backbone = getattr(smp, decoder_type)(
            encoder_name=encoder_type,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=len(
                config["dataset"]["band_names"]
            ),  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=out_ch,
        )  # model output channels (number of classes in your dataset))

        # For timm models, we can extract the mean and std of the pre-trained backbone
        # hparams.update({"mean": backbone.default_cfg["mean"]})
        # hparams.update({"std": backbone.default_cfg["std"]})
        config["model"]["input_size"] = (
            len(config["dataset"]["band_names"]),
            config["model"]["image_size"],
            config["model"]["image_size"],
        )

        with torch.no_grad():
            backbone.eval()
            features = torch.zeros(config["model"]["input_size"]).unsqueeze(0)
            features = backbone.encoder(features)

        class Noop(torch.nn.Module):
            def forward(self, x):
                return x

        head = Noop()  # pytorch image models already adds a classifier on top of the UNETs
        # head = head_generator(task_specs, shapes, hparams)
        loss = train_loss_generator(task_specs, config)
        train_metrics = train_metrics_generator(task_specs, config)
        eval_metrics = eval_metrics_generator(task_specs, config)
        return Model(backbone, head, loss, config, train_metrics, eval_metrics)

    def get_collate_fn(self, task_specs: TaskSpecifications, config: dict):
        """Define a collate function to batch input tensors.

        Args:
            task_specs: task specs to retrieve dataset
            hparams: model hyperparameters

        Returns:
            collate function
        """
        return default_collate

    def get_transform(
        self,
        task_specs: TaskSpecifications,
        config: Dict[str, Any],
        train=True,
        scale=None,
        ratio=None,
    ):
        """Define data transformations specific to the models generated.

        Args:
            task_specs: task specs to retrieve dataset
            hparams: model hyperparameters
            train: train mode true or false
            scale: define image scale
            ratio: define image ratio range

        Returns:
            callable function that applies transformations on input data
        """
        scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
        ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
        c, h, w = config["model"]["input_size"]
        mean, std = task_specs.get_dataset(
            split="train",
            format=config["dataset"]["format"],
            band_names=tuple(config["dataset"]["band_names"]),
            benchmark_dir=config["experiment"]["benchmark_dir"],
            partition_name=config["experiment"]["partition_name"],
        ).rgb_stats()
        band_names = tuple(config["dataset"]["band_names"])

        class SegTransform:
            """Segmentation Transform.

            This is a helper class for applying the same transformation
            to input images and segmentation masks.
            """

            def __call__(self, x: Tensor, resample: bool = True, train: bool = True):
                """Apply data augmentation to input and segmentation mask.

                Args:
                    x: input image or segmentation mask
                    resample: whether to resample (True) or reuse (False) previous transforms.
                                               Defaults to True.
                    train: whether in training mode. No aug during validation. Defaults to True.

                Returns:
                    transformed input image or segmentation mask
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

            x = sample.pack_to_3d(band_names=band_names)[0].astype("float32")
            x, y = t_x(x), t_y(sample.label.data.astype("float32"))
            return {"input": x, "label": y.long().squeeze()}

        return transform


def model_generator() -> SegmentationGenerator:
    """Initialize Segmentation Generator.

    Returns:
        segmentation model generator
    """
    return SegmentationGenerator()
