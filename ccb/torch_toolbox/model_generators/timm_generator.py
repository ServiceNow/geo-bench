from typing import List
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


class TIMMGenerator(ModelGenerator):
    def __init__(self, hparams=None) -> None:
        super().__init__()

        self.base_hparams = {
            "backbone": "resnet18",
            "pretrained": True,
            "lr_backbone": 1e-4,
            "lr_head": 1e-3,
            "optimizer": "sgd",
            "head_type": "linear",
            "loss_type": "crossentropy",
            "batch_size": 128,
            "num_workers": 4,
            "max_epochs": 10,
            "n_gpus": 1,
            "logger": "wandb",
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
        backbone = timm.create_model(
            hyperparameters["backbone"], pretrained=hyperparameters["pretrained"], features_only=True
        )
        logging.warn("FIXME: Using ImageNet default input size, mean, and std!")
        hyperparameters.update({"input_size": backbone.default_cfg["input_size"]})
        hyperparameters.update({"mean": backbone.default_cfg["mean"]})
        hyperparameters.update({"std": backbone.default_cfg["std"]})
        features = torch.zeros(hyperparameters["input_size"]).unsqueeze(0)
        features = backbone(features)
        shapes = [x.shape[1:] for x in features]
        hyperparameters.update({"features_shape": shapes})

        head = head_generator(task_specs, hyperparameters)
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
        mean = hyperparams["mean"]
        std = hyperparams["std"]
        t = []
        t.append(tt.Lambda(lambda x: x.pack_to_3d(band_names=("red", "green", "blue"))[0].astype("float32")))
        t.append(tt.Lambda(lambda x: x / x.max()))
        t.append(tt.ToTensor())
        t.append(tt.Normalize(mean=mean, std=std))
        if train:
            t.append(tt.RandomHorizontalFlip())
            t.append(tt.RandomResizedCrop((h, w), scale=scale, ratio=ratio))
        t = tt.Compose(t)
        return t


model_generator = TIMMGenerator()
