from typing import List
import torch
from pytorch_lightning import LightningModule
from ccb import io
import numpy as np
import torch.nn.functional as F
import torchmetrics
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.modules import ClassificationHead


class Model(LightningModule):
    """
    Default Model class provided by the toolbox.

    TODO(pau-pow)
    """

    def __init__(self, backbone, head, loss_function, hyperparameters, train_metrics=None, eval_metrics=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.loss_function = loss_function
        self.train_metrics = train_metrics or (lambda *args: {})
        self.eval_metrics = eval_metrics or (lambda *args: {})
        # Make TorchMetrics visible to Torchvision
        self._train_metrics = self.train_metrics.get_torchmetrics()
        self._eval_metrics = self.eval_metrics.get_torchmetrics()
        self.hyperparameters = hyperparameters
        self.save_hyperparameters("hyperparameters")

    def forward(self, x):
        if self.hyperparameters["lr_backbone"] == 0:
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        logits = self.head(features)
        return logits

    def training_step(self, batch, batch_idx):
        inputs = batch["input"]
        target = batch["label"]
        output = self(inputs)
        loss_train = self.loss_function(output, target)
        return {"loss": loss_train, "output": output.detach(), "target": target.detach()}

    def training_step_end(self, outputs):
        # update and log
        metrics = self.train_metrics(outputs["output"], outputs["target"], "train")
        self.log("train_loss", outputs["loss"], logger=True)
        self.log_dict(metrics)

    def eval_step(self, batch, batch_idx, prefix):
        inputs = batch["input"]
        target = batch["label"]
        output = self(inputs)
        loss = self.loss_function(output, target)
        self.log(f"{prefix}_loss", loss, logger=True, prog_bar=True)  # , on_step=True, on_epoch=True, logger=True)
        metrics = self.eval_metrics(output, target, prefix)
        self.log_dict(metrics, logger=True)
        return {"loss": loss.detach(), "output": output.detach(), "target": target.detach()}

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def eval_step_end(self, outputs, prefix):
        # update and log
        metrics = self.eval_metrics(outputs["output"], outputs["target"], prefix)
        self.log(f"{prefix}_loss", outputs["loss"], logger=True)
        self.log_dict(metrics)

    def validation_step_end(self, outputs):
        self.eval_step_end(outputs, "val")

    def test_step_end(self, outputs):
        self.eval_step_end(outputs, "test")

    def configure_optimizers(self):
        backbone_parameters = self.backbone.parameters()
        backbone_parameters = list(filter(lambda p: p.requires_grad, backbone_parameters))
        head_parameters = self.head.parameters()
        head_parameters = list(filter(lambda p: p.requires_grad, head_parameters))
        lr_backbone = self.hyperparameters["lr_backbone"]
        lr_head = self.hyperparameters["lr_head"]
        momentum = self.hyperparameters.get("momentum", 0.9)
        nesterov = self.hyperparameters.get("nesterov", True)
        weight_decay = self.hyperparameters.get("weight_decay", 1e-4)
        optimizer_type = self.hyperparameters.get("optimizer", "sgd").lower()
        to_optimize = []
        for params, lr in [(backbone_parameters, lr_backbone), (head_parameters, lr_head)]:
            if lr > 0:
                to_optimize.append({"params": params, "lr": lr})
        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                to_optimize,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(to_optimize)
        elif optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(to_optimize, weight_decay=weight_decay)

        if self.hyperparameters.get("scheduler", None) == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.hyperparameters["lr_milestones"], gamma=self.hyperparameters["lr_gamma"]
            )
            return [optimizer], [scheduler]
        else:
            scheduler = None
            return [optimizer]


class ModelGenerator:
    """
    Class implemented by the user. The goal is to specify how to connect the backbone with the head and the loss function.
    """

    def __init__(self, model_path=None) -> None:
        """This should not load the model at this point"""
        self.model_path = model_path

    def hp_search(self, task_specs, max_num_configs=10):
        """The user can provide a set of `max_num_configs` hyperparameters configuration to search for, based on task_specs"""
        # hp_configs = [dict(lr=0.4, width=100), dict(lr=0.1, width=100), dict(lr=0.1, width=200)]
        # return hparams_to_string(hp_configs)
        raise NotImplementedError()

    def generate(self, task_specs, hyperparams):
        """Generate a Model to train

        Args:
            task_specs (TaskSpecifications): an object describing the task to be performed
            hyperparams (dict): dictionary containing hyperparameters of the experiment

        Raises:
            NotImplementedError

        Example:
            backbone = MyBackBone(self.model_path, task_specs, hyperparams) # Implemented by the user so that he can wrap his
            head = head_generator(task_specs, hyperparams) # provided by the toolbox or the user can implement their own
            loss = train_loss_generator(task_specs, hyperparams) # provided by the toolbox or the user can implement their own
            return Model(backbone, head, loss, hyperparams) # base model provided by the toolbox
        """
        raise NotImplementedError()

    def get_collate_fn(self, task_specs, hyperparams):
        """Generate the collate functions for stacking the mini-batch.

        Args:
            task_specs (TaskSpecifications): an object describing the task to be performed
            hyperparams (dict): dictionary containing hyperparameters of the experiment

        Returns:
            A callable mapping a list of Sample to a tuple containing stacked inputs and labels. The stacked inputs
            will be fed to the model.

        Raises:
            NotImplementedError

        Example:
            return ccb.torch_toolbox.model.collate_rgb
        """
        raise None

    def get_transform(self, task_specs, hyperparams, train=True):
        """Generate the collate functions for stacking the mini-batch.

        Args:
            task_specs (TaskSpecifications): an object describing the task to be performed
            hyperparams (dict): dictionary containing hyperparameters of the experiment

        Returns:
            A callable taking an object of type Sample as input. The return will be fed to the collate_fn
        """
        return None


def head_generator(task_specs: TaskSpecifications, features_shape: List[tuple], hyperparams: dict):
    """
    Returns a an appropriate head based on the task specifications. We can use task_specs.task_type as follow:
        classification: 2 layer MLP with softmax activation
        semantic_segmentation: U-Net decoder.
    we can also do something special for a specific dataet using task_specs.dataset_name. Hyperparams and input_shape
    can also be used to adapt the head.

    Args:
        task_specs: object of type TaskSpecifications providing information on what type of task we are solving
        features_shape: lists with the shapes of the output features at different depths in the architecture [(ch, h, w), ...]
        hyperparams: dict of hyperparameters.
    """
    if isinstance(task_specs.label_type, io.Classification):
        if hyperparams["head_type"] == "linear":
            in_ch, *other_dims = features_shape[-1]
            out_ch = task_specs.label_type.n_classes
            return ClassificationHead(in_ch, out_ch)
        else:
            raise ValueError(f"Unrecognized head type: {hyperparams['head_type']}")
    else:
        raise ValueError(f"Unrecognized task: {task_specs.task_type}")


def vit_head_generator(task_specs, hyperparams, input_shape):
    """
    ViT architectures may require different type of heads. In which case, we should provide this to the users as well. TO BE DISCUSSED.
    """
    pass


def compute_accuracy(output, target, prefix, topk=(1,), *args, **kwargs):
    """Computes the accuracy over the k top predictions for the specified values of k."""
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


def train_metrics_generator(task_specs: io.TaskSpecifications, hparams: dict):
    """
    Returns the appropriate loss function depending on the task_specs. We should implement basic loss and we can leverage the
    following attributes: task_specs.task_type and task_specs.eval_loss
    """

    metrics = {
        io.Classification: [
            torchmetrics.Accuracy(dist_sync_on_step=True),
        ],
        io.SegmentationClasses: [],
    }[task_specs.label_type.__class__]

    for metric_name in hparams.get("train_metrics", ()):
        metrics.append(METRIC_MAP[metric_name])

    # The Metrics().__call__(input, targets) accumulates all metrics in a dict
    return Metrics(metrics)


def eval_metrics_generator(task_specs: io.TaskSpecifications, hparams: dict):
    """
    Returns the appropriate eval function depending on the task_specs.
    """
    metrics = {
        io.Classification: [
            torchmetrics.Accuracy(),
        ],
        io.SegmentationClasses: (),
    }[task_specs.label_type.__class__]

    for metric_name in hparams.get("eval_metrics", ()):
        metrics.append(METRIC_MAP[metric_name])

    return Metrics(metrics)


def train_loss_generator(task_specs: io.TaskSpecifications, hparams):
    """
    Returns the appropriate loss function depending on the task_specs.
    """
    loss = {io.Classification: F.cross_entropy, io.SegmentationClasses: F.cross_entropy}[
        task_specs.label_type.__class__
    ]

    return loss


class Metrics:
    def __init__(self, metrics: List):
        super().__init__()
        self.metrics = metrics

    def add_metric(self, metric):
        self.metrics.append(metric)

    def get_torchmetrics(self):
        ret = []
        for metric in self.metrics:
            if isinstance(metric, torchmetrics.Metric):
                ret.append(metric)
        return torchmetrics.MetricCollection(ret)

    def __call__(self, output, target, prefix="", *args, **kwargs) -> dict:
        ret = {}
        for metric in self.metrics:
            if isinstance(metric, torchmetrics.Accuracy):
                # metric.to(output.device)
                ret["_".join([prefix, "accuracy-1"])] = metric(output, target)
            else:
                ret.update(metric(output, target, prefix, *args, **kwargs))
        return ret


class BackBone(torch.nn.Module):
    def __init__(self, model_path, task_specs, hyperparams) -> None:
        super().__init__()
        self.model_path = model_path
        self.task_specs = task_specs
        self.hyperparams = hyperparams

    def forward(self, data_dict):
        """
        data_dict is a collection of tensors returned by the data loader.
        The user is responsible to implement something that will map
        the information from the dataset and encode it into a list of tensors.
        Returns: the encoded representation or a list of representations for
        models like u-net.
        raise NotImplementedError()
        """


def collate_rgb(samples: List[io.Sample]):
    x_list = []
    label_list = []
    for sample in samples:
        rgb_image, _ = sample.pack_to_3d(band_names=("red", "green", "blue"))
        x_list.append(torch.from_numpy(np.moveaxis(rgb_image.astype(np.float32), 2, 0)))
        label_list.append(sample.label)

    return torch.stack(x_list), stack_labels(label_list)


def stack_labels(label_list):
    if isinstance(label_list[0], int):
        return torch.tensor(label_list)
    else:
        raise NotImplementedError()
