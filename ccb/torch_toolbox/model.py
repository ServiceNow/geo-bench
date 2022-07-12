"""Model."""

import os
import random
import string
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import Tensor

from ccb import io
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.modules import ClassificationHead


class Model(LightningModule):
    """Default Model class provided by the toolbox.

    Define model training, evaluation and testing steps for
    `Pytorch LightningModule <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html>`_
    """

    def __init__(
        self,
        backbone,
        head: ClassificationHead,
        loss_function,
        config,
        train_metrics=None,
        eval_metrics=None,
    ) -> None:
        """Initialize a new instance of Model.

        Args:
            backbone: model backbone
            head: model prediction head
            loss_function: loss function for training
            hyperparameters: model hyperparameters
            train_metrics: metrics used during training
            eval_metrics: metrics used during evaluation

        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.loss_function = loss_function
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self.config = config

    def forward(self, x):
        """Forward input through model.

        Args:
            x: input

        Returns:
            feature representation
        """
        if self.config["model"]["lr_backbone"] == 0:
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        logits = self.head(features)
        return logits

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        """Define steps taken during training mode.

        Args:
            batch: input batch
            batch_idx: index of batch

        Returns:
            training step outputs
        """
        inputs = batch["input"]
        target = batch["label"]
        output = self(inputs)
        loss_train = self.loss_function(output, target)
        return {"loss": loss_train, "output": output.detach(), "target": target.detach()}

    def training_step_end(self, outputs: Dict[str, Tensor]) -> None:
        """Define steps taken after training phase.

        Args:
            outputs: outputs from :meth:`__training_step`
        """
        # update and log
        self.log("train_loss", outputs["loss"], logger=True)
        self.train_metrics.update(outputs["output"], outputs["target"])

    def training_epoch_end(self, outputs: Dict[str, Tensor]) -> None:
        """Define actions after a training epoch.

        Args:
            outputs: outputs from :meth:`__training_step`
        """
        self.log_dict({f"train_{k}": v for k, v in self.train_metrics.compute().items()}, logger=True)
        self.train_metrics.reset()

    def eval_step(self, batch: Tensor, batch_idx: int, prefix: str) -> Dict[str, Tensor]:
        """Define steps taken during validation and testing.

        Args:
            batch: input batch
            batch_idx: index of batch
            prefix: prefix for logging

        Returns:
            evaluation step outputs
        """
        inputs = batch["input"]
        target = batch["label"]
        output = self(inputs)
        loss = self.loss_function(output, target)
        self.log(f"{prefix}_loss", loss, logger=True, prog_bar=True)  # , on_step=True, on_epoch=True, logger=True)

        return {"loss": loss.detach(), "input": inputs.detach(), "output": output.detach(), "target": target.detach()}

    def validation_step(self, batch, batch_idx):
        """Define steps taken during validation mode.

        Args:
            batch: input batch
            batch_idx: index of batch

        Returns:
            validation step outputs
        """
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """Define steps taken during test mode.

        Args:
            batch: input batch
            batch_idx: index of batch

        Returns:
            test step outputs
        """
        return self.eval_step(batch, batch_idx, "test")

    def eval_step_end(self, outputs, prefix) -> None:
        """Define steps after evaluation phase.

        Args:
            outputs: outputs from :meth:`__eval_step`
            prefix: prefix for logging
        """
        # update and log
        self.log(f"{prefix}_loss", outputs["loss"], logger=True)
        self.eval_metrics.update(outputs["output"], outputs["target"])

    def validation_step_end(self, outputs):
        """Define steps after validation phase.

        Args:
            outputs: outputs from :meth:`__eval_step`
        """
        self.eval_step_end(outputs, "val")

    def test_step_end(self, outputs):
        """Define steps after testing phase.

        Args:
            outputs: outputs from :meth:`__eval_step`
        """
        self.eval_step_end(outputs, "test")

    def validation_epoch_end(self, outputs):
        """Define actions after a validation epoch.

        Args:
            outputs: outputs from :meth:`__validation_step`
        """
        eval_metrics = self.eval_metrics.compute()
        self.log_dict({f"val_{k}": v for k, v in eval_metrics.items()}, logger=True)
        self.eval_metrics.reset()
        if self.config["model"].get("log_segmentation_masks", False):
            import wandb

            current_element = int(torch.randint(0, outputs[0]["input"].shape[0], size=(1,)))
            image = outputs[0]["input"][current_element].permute(1, 2, 0).cpu().numpy()
            pred_mask = outputs[0]["output"].argmax(1)[current_element].cpu().numpy()
            gt_mask = outputs[0]["target"][current_element].cpu().numpy()
            image = wandb.Image(
                image, masks={"predictions": {"mask_data": pred_mask}, "ground_truth": {"mask_data": gt_mask}}
            )
            wandb.log({"segmentation_images": image})

    def test_epoch_end(self, outputs):
        """Define actions after a test epoch.

        Args:
            outputs: outputs from :meth:`__test_step`
        """
        eval_metrics = self.eval_metrics.compute()
        self.log_dict({f"test_{k}": v for k, v in eval_metrics.items()}, logger=True)
        self.eval_metrics.reset()

    def configure_optimizers(self):
        """Configure optimizers for training."""
        backbone_parameters = self.backbone.parameters()
        # backbone_parameters = list(filter(lambda p: p.requires_grad, backbone_parameters))
        head_parameters = self.head.parameters()
        # head_parameters = list(filter(lambda p: p.requires_grad, head_parameters))
        lr_backbone = self.config["model"]["lr_backbone"]
        lr_head = self.config["model"]["lr_head"]
        momentum = self.config["model"].get("momentum", 0.9)
        nesterov = self.config["model"].get("nesterov", True)
        weight_decay = self.config["model"].get("weight_decay", 1e-4)
        optimizer_type = self.config["model"].get("optimizer", "sgd").lower()
        to_optimize = []
        for params, lr in [(backbone_parameters, lr_backbone), (head_parameters, lr_head)]:
            if lr > 0:
                to_optimize.append({"params": params, "lr": lr})
        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(to_optimize, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(to_optimize)
        elif optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(to_optimize, weight_decay=weight_decay)

        if self.config["model"].get("scheduler", None) == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.config["model"]["lr_milestones"], gamma=self.config["model"]["lr_gamma"]
            )
            return [optimizer], [scheduler]
        else:
            scheduler = None
            return [optimizer]


class ModelGenerator:
    """Model Generator.

    Class implemented by the user. The goal is to specify how to connect the backbone with the head and the loss function.
    """

    def __init__(self, model_path=None) -> None:
        """Initialize a new instance of Model Generator.

        This should not load the model at this point

        Args:
            model_path: path to model

        """
        self.model_path = model_path

    def generate_model(self, task_specs: TaskSpecifications, config: Dict[str, Any]):
        """Generate a Model to train.

        Args:
            task_specs: an object describing the task to be performed
            config: config dictionary containing experiment and hyperparameters configurations

        Raises:
            NotImplementedError

        Example:
            backbone = MyBackBone(self.model_path, task_specs, hyperparams) # Implemented by the user so that he can wrap his
            head = head_generator(task_specs, hyperparams) # provided by the toolbox or the user can implement their own
            loss = train_loss_generator(task_specs, hyperparams) # provided by the toolbox or the user can implement their own
            return Model(backbone, head, loss, hyperparams) # base model provided by the toolbox
        """
        raise NotImplementedError()

    def generate_trainer(self, config: dict, job) -> pl.Trainer:
        """Configure a pytroch lightning Trainer.

        Args:
            config: dictionary containing config
            job: job being executed to let logger know directory

        Returns:
            lightning Trainer with configurations from config file.
        """
        run_id = "".join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(8))
        config["wandb"]["wandb_run_id"] = run_id

        loggers = [
            pl.loggers.CSVLogger(str(job.dir), name="lightning_logs"),
            pl.loggers.WandbLogger(
                save_dir=str(job.dir),
                project=config["wandb"]["project"],
                entity=config["wandb"]["entity"],
                id=run_id,
                group=config["wandb"].get("wandb_group", None),
                name=config["wandb"].get("name", None),
                resume="allow",
                config=config["model"],
            ),
        ]

        job.save_config(config, overwrite=True)

        ckpt_dir = os.path.join(job.dir, "checkpoint")
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir, save_top_k=1, monitor="val_loss", mode="min", every_n_epochs=1
        )

        trainer = pl.Trainer(
            **config["pl"],
            default_root_dir=job.dir,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", mode="min", patience=config["pl"].get("patience", 30), min_delta=1e-5
                ),
                checkpoint_callback,
            ],
            logger=loggers,
        )

        return trainer

    def get_collate_fn(self, task_specs: TaskSpecifications, config: Dict[str, Any]):
        """Generate the collate functions for stacking the mini-batch.

        Args:
            task_specs: an object describing the task to be performed
            config: dictionary containing hyperparameters of the experiment

        Returns:
            A callable mapping a list of Sample to a tuple containing stacked inputs and labels. The stacked inputs
            will be fed to the model.

        Raises:
            NotImplementedError

        Example:
            return ccb.torch_toolbox.model.collate_rgb
        """
        raise NotImplementedError()

    def get_transform(self, task_specs: TaskSpecifications, config: Dict[str, Any], train: bool = True):
        """Generate the collate functions for stacking the mini-batch.

        Args:
            task_specs: an object describing the task to be performed
            hyperparams: dictionary containing hyperparameters of the experiment
            train: whether to return train or evaluation transforms

        Returns:
            A callable taking an object of type Sample as input. The return will be fed to the collate_fn
        """
        raise NotImplementedError()


def head_generator(task_specs: TaskSpecifications, features_shape: List[tuple], config: Dict[str, Any]):
    """Return an appropriate head based on the task specifications.

    We can use task_specs.task_type as follow:
        classification: 2 layer MLP with softmax activation
        semantic_segmentation: U-Net decoder.
    we can also do something special for a specific dataet using task_specs.dataset_name. Hyperparams and input_shape
    can also be used to adapt the head.

    Args:
        task_specs: providing information on what type of task we are solving
        features_shape: lists with the shapes of the output features at different depths in the architecture [(ch, h, w), ...]
        hyperparams: dict of hyperparameters.

    """
    if isinstance(task_specs.label_type, io.Classification):
        if config["model"]["head_type"] == "linear":
            in_ch, *other_dims = features_shape[-1]
            out_ch = task_specs.label_type.n_classes
            return ClassificationHead(in_ch, out_ch, hidden_size=config["model"]["hidden_size"])
        else:
            raise ValueError(f"Unrecognized head type: {config['model']['head_type']}")
    elif isinstance(task_specs.label_type, io.MultiLabelClassification):
        if config["model"]["head_type"] == "linear":
            in_ch, *other_dims = features_shape[-1]
            out_ch = task_specs.label_type.n_classes
            return ClassificationHead(in_ch, out_ch, hidden_size=config["model"]["hidden_size"])
        else:
            raise ValueError(f"Unrecognized head type: {config['model']['head_type']}")
    elif isinstance(task_specs.label_type, io.SemanticSegmentation):
        if config["model"]["head_type"].split("-")[0] == "smp":  # smp: segmentation-models-pytorch
            return lambda *args: args
        else:
            raise ValueError(f"Unrecognized head type: {config['model']['head_type']}")
    else:
        raise ValueError(f"Unrecognized task: {task_specs.label_type}")


def vit_head_generator(task_specs: TaskSpecifications, config: Dict[str, Any], input_shape: int):
    """Generate head for VIT.

    ViT architectures may require different type of heads.
    In which case, we should provide this to the users as well. TO BE DISCUSSED.

    Args:
        task_specs: an object describing the task to be performed
        hyperparams: dictionary containing hyperparameters of the experiment
        input_shape: input shape to transformer

    """
    pass


def compute_accuracy(
    output: Tensor, target: Tensor, prefix: str, topk: Tuple[int] = (1,), *args, **kwargs
) -> Dict[str, float]:
    """Compute the accuracy over the k top predictions for the specified values of k.

    Args:
        output: model output
        target: target to compute accuracy on
        pefix: prefix for k
        topk: define k values for which to compute accuracy
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        computed accuracy values for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f"{prefix}_accuracy-{k}"] = correct_k.mul_(100.0 / batch_size)
        return res


METRIC_MAP = {}


def train_metrics_generator(task_specs: TaskSpecifications, config: Dict[str, Any]) -> torchmetrics.MetricCollection:
    """Return the appropriate loss function depending on the task_specs.

    We should implement basic loss and we can leverage the
    following attributes: task_specs.task_type and task_specs.eval_loss

    Args:
        task_specs: an object describing the task to be performed
        config: dictionary containing hyperparameters of the experiment

    Returns:
        metric collection used during training
    """
    metrics = {
        io.Classification: [torchmetrics.Accuracy(dist_sync_on_step=True, top_k=1)],
        io.MultiLabelClassification: [
            # torchmetrics.Accuracy(dist_sync_on_step=True),
            torchmetrics.F1Score(task_specs.label_type.n_classes)
        ],
        io.SegmentationClasses: [torchmetrics.JaccardIndex(task_specs.label_type.n_classes)],
    }[task_specs.label_type.__class__]

    for metric_name in config["model"].get("train_metrics", ()):
        metrics.extend(METRIC_MAP[metric_name])

    return torchmetrics.MetricCollection(metrics)


def eval_metrics_generator(task_specs: TaskSpecifications, config: Dict[str, Any]) -> torchmetrics.MetricCollection:
    """Return the appropriate eval function depending on the task_specs.

    Args:
        task_specs: an object describing the task to be performed
        hyperparams: dictionary containing hyperparameters of the experiment

    Returns:
        metric collection used during evaluation
    """
    metrics = {
        io.Classification: [torchmetrics.Accuracy()],
        io.SegmentationClasses: [
            torchmetrics.JaccardIndex(task_specs.label_type.n_classes),
            torchmetrics.FBetaScore(task_specs.label_type.n_classes, beta=2, mdmc_average="samplewise"),
        ],
        io.MultiLabelClassification: [torchmetrics.F1Score(task_specs.label_type.n_classes)],
    }[task_specs.label_type.__class__]

    for metric_name in config["model"].get("eval_metrics", ()):
        metrics.extend(METRIC_MAP[metric_name])

    return torchmetrics.MetricCollection(metrics)


def _balanced_binary_cross_entropy_with_logits(outputs: Tensor, targets: Tensor) -> Tensor:
    """Compute balance binary cross entropy for multi-label classification.

    Args:
        outputs: model outputs
        targets: targets to compute binary cross entropy on
    """
    classes = targets.shape[-1]
    outputs = outputs.view(-1, classes)
    targets = targets.view(-1, classes).float()
    loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    loss = loss[targets == 0].mean() + loss[targets == 1].mean()
    return loss


def train_loss_generator(task_specs: TaskSpecifications, config: Dict[str, Any]) -> Dict:
    """Return the appropriate loss function depending on the task_specs.

    Args:
        task_specs: an object describing the task to be performed
        config: dictionary containing hyperparameters of the experiment

    Returns:
        available loss functions for training
    """
    loss = {
        io.Classification: F.cross_entropy,
        io.MultiLabelClassification: _balanced_binary_cross_entropy_with_logits,
        io.SegmentationClasses: F.cross_entropy,
    }[task_specs.label_type.__class__]

    return loss


class BackBone(torch.nn.Module):
    """Backbone.

    Create a model Backbone to produce feature representations.

    """

    def __init__(self, model_path: str, task_specs: TaskSpecifications, config: Dict[str, Any]) -> None:
        """Initialize a new instance of Backbone.

        Args:
            model_path:
            task_specs: an object describing the task to be performed
            config: dictionary containing hyperparameters of the experiment

        """
        super().__init__()
        self.model_path = model_path
        self.task_specs = task_specs
        self.config = config

    def forward(self, data_dict: Dict[str, Tensor]) -> Tensor:
        """Forward input through backbone.

        Args:
            data_dict:  is a collection of tensors returned by the data loader.

        Returns:
            the encoded representation or a list of representations for
        """


def collate_rgb(samples: List[io.Sample]) -> Dict[str, Tensor]:
    """Collate function for RGB images.

    Args:
        samples: list of samples

    Returns:
        collated version of samples
    """
    x_list = []
    label_list = []
    for sample in samples:
        rgb_image, _ = sample.pack_to_3d(band_names=("red", "green", "blue"))
        x_list.append(torch.from_numpy(np.moveaxis(rgb_image.astype(np.float32), 2, 0)))
        label_list.append(sample.label)

    return {"input": torch.stack(x_list), "label": stack_labels(label_list)}


def stack_labels(label_list: List[Tensor]) -> Tensor:
    """Stack labels for collate function.

    Args:
        label_list: list of labels to be stacked

    Returns:
        stacked labels
    """
    if isinstance(label_list[0], int):
        return torch.tensor(label_list)
    else:
        raise NotImplementedError()
