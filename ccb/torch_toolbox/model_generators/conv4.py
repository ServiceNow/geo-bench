from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as tt

from ccb import io
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.model import (
    BackBone,
    Model,
    ModelGenerator,
    collate_rgb,
    eval_metrics_generator,
    head_generator,
    train_loss_generator,
    train_metrics_generator,
)


class Conv4Generator(ModelGenerator):
    def __init__(self, hparams=None) -> None:
        super().__init__()

        self.base_hparams = {
            "backbone": "conv4",
            "lr_gamma": 0.1,
            "lr_backbone": 4e-3,
            "lr_head": 4e-3,
            "head_type": "linear",
            "hidden_size": 128,
            "loss_type": "crossentropy",
            "batch_size": 32,
            "num_workers": 0,
            "max_epochs": 1,
            "n_gpus": 0,
            "logger": "csv",
            "sweep_config_yaml_path": "/mnt/home/climate-change-benchmark/ccb/torch_toolbox/wandb/hparams_classification_conv4.yaml",
            "num_seeds": 3,
            "num_agents": 4,
            "num_trials_per_agent": 5,
            "band_names": ["red", "green", "blue"],
            "image_size": 224,
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
        backbone = Conv4(self.model_path, task_specs, hyperparameters)
        head = head_generator(task_specs, [(64,)], hyperparameters)
        loss = train_loss_generator(task_specs, hyperparameters)
        train_metrics = train_metrics_generator(task_specs, hyperparameters)
        eval_metrics = eval_metrics_generator(task_specs, hyperparameters)
        return Model(backbone, head, loss, hyperparameters, train_metrics, eval_metrics)

    def get_collate_fn(self, task_specs: TaskSpecifications, hparams: dict):
        return default_collate

    def get_transform(self, task_specs, hyperparams, train=True, scale=None, ratio=None):
        scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
        ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
        _, h, w = (len(hyperparams["band_names"]), hyperparams["image_size"], hyperparams["image_size"])

        if task_specs.dataset_name == "imagenet":
            mean, std = task_specs.get_dataset(split="train", format=hyperparams["format"]).rgb_stats()
            t = []
            t.append(tt.ToTensor())
            t.append(tt.Normalize(mean=mean, std=std))
            if train:
                t.append(tt.RandomHorizontalFlip())
                t.append(tt.RandomResizedCrop((h, w), scale=scale, ratio=ratio))
            transform = tt.Compose(t)
        elif task_specs.dataset_name.lower() == "mnist":
            t = []
            t.append(tt.ToTensor())
            transform = tt.Compose(t)
        else:
            mean, std = task_specs.get_dataset(
                split="train", format=hyperparams["format"], band_names=tuple(hyperparams["band_names"])
            ).normalization_stats()
            t = []
            t.append(tt.ToTensor())
            t.append(tt.Normalize(mean=mean, std=std))
            if train:
                t.append(tt.RandomHorizontalFlip())
                t.append(tt.RandomResizedCrop((h, w), scale=scale, ratio=ratio))

            t.append(tt.Resize((hyperparams["image_size"], hyperparams["image_size"])))

            t = tt.Compose(t)

            def transform(sample: io.Sample):
                x = sample.pack_to_3d(band_names=tuple(hyperparams["band_names"]))[0].astype("float32")
                x = t(x)
                return {"input": x, "label": sample.label}

        return transform


def model_generator(hparams: Dict[str, Any] = {}) -> Conv4Generator:
    model_generator = Conv4Generator(hparams=hparams)
    return model_generator


class Conv4(BackBone):
    def __init__(self, model_path, task_specs: io.TaskSpecifications, hyperparams):
        super().__init__(model_path, task_specs, hyperparams)
        n_bands = min(3, len(task_specs.bands_info))
        self.conv0 = torch.nn.Conv2d(n_bands, 64, 3, 1, 1)
        self.conv1 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv0(x), True)
        x = F.relu(self.conv1(x), True)
        x = F.max_pool2d(x, 3, 2, 1)
        x = F.relu(self.conv2(x), True)
        x = F.max_pool2d(x, 3, 2, 1)
        x = F.relu(self.conv3(x), True)
        x = F.max_pool2d(x, 3, 2, 1)
        return x.mean((2, 3))
