import torch
import torchvision.transforms as tt
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from ccb.io import Classification, Accuracy, CrossEntropy
from typing import List


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
        self.hyperparameters = hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_train = self.loss_function(output, target)
        metrics = self.train_metrics(output, target, "train")
        self.log("train_loss", loss_train, logger=True)
        self.log_dict(metrics)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix):
        images, target = batch
        output = self(images)
        loss = self.loss_function(output, target)
        self.log(f"{prefix}_loss", loss, logger=True, prog_bar=True)  # , on_step=True, on_epoch=True, logger=True)
        # acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        metrics = self.eval_metrics(output, target, prefix)
        self.log_dict(metrics, logger=True)
        # self.log(
        #     f"{prefix}_acc1", acc1, prog_bar=True, logger=True
        # )  # on_step=True, prog_bar=True, on_epoch=True, logger=True)
        # self.log(f"{prefix}_acc5", acc5, logger=True)  # , on_step=True, on_epoch=True, logger=True)
        return metrics

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        backbone_parameters = self.backbone.parameters()
        backbone_parameters = list(filter(lambda p: p.requires_grad, backbone_parameters))
        head_parameters = self.head.parameters()
        head_parameters = list(filter(lambda p: p.requires_grad, head_parameters))
        lr_backbone = self.hyperparameters["lr_backbone"]
        lr_head = self.hyperparameters["lr_head"]
        optimizer = torch.optim.Adam(
            [{"params": backbone_parameters, "lr": lr_backbone}, {"params": head_parameters, "lr": lr_head}]
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.hyperparameters["lr_milestones"], gamma=self.hyperparameters["lr_gamma"]
        )
        return [optimizer], [scheduler]


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


def head_generator(task_specs, hyperparams):
    """
    Returns a an appropriate head based on the task specifications. We can use task_specs.task_type as follow:
        classification: 2 layer MLP with softmax activation
        semantic_segmentation: U-Net decoder.
    we can also do something special for a specific dataet using task_specs.dataset_name. Hyperparams and input_shape
    can also be used to adapt the head.

    Args:
        task_specs: object of type TaskSpecifications providing information on what type of task we are solving
        hyperparams: dict of hyperparameters.
        input_shape: list of tuples describing the shape of the input of this module. TO BE DISCUSSED: should this be
            the input itself? should it be a dict of shapes?
    """
    if isinstance(task_specs.label_type, Classification):
        if hyperparams["head_type"] == "linear":
            (in_ch,) = hyperparams["features_shape"]
            out_ch = task_specs.label_type.n_classes
            return torch.nn.Linear(in_ch, out_ch)
        else:
            raise ValueError(f"Unrecognized head type: {hyperparams['head_type']}")
    else:
        raise ValueError(f"Unrecognized task: {task_specs.task_type}")


def vit_head_generator(task_specs, hyperparams, input_shape):
    """
    ViT architectures may require different type of heads. In which case, we should provide this to the users as well. TO BE DISCUSSED.
    """
    pass


def train_loss_generator(task_specs, hyperparams):
    """
    Returns the appropriate loss function depending on the task_specs. We should implement basic loss and we can leverage the
    following attributes: task_specs.task_type and task_specs.eval_loss
    """
    if isinstance(task_specs.label_type, Classification):
        if hyperparams["loss_type"] == "crossentropy":
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unrecognized loss type: {hyperparams['head_type']}")
    else:
        raise ValueError(f"Unrecognized task: {task_specs.label_type}")


class Metrics:
    def __init__(self, metrics: List):
        self.metrics = metrics

    def add_metric(self, metric):
        self.metrics.append(metric)

    def __call__(self, output, target, prefix, *args, **kwargs) -> dict:
        ret = {}
        for metric in self.metrics:
            if isinstance(metric, Accuracy):
                ret.update(compute_accuracy(output, target, prefix))
            elif isinstance(metric, CrossEntropy):
                loss = F.cross_entropy(output, target, *args, **kwargs)
                ret[f"{prefix}_loss"] = loss
        return ret


def eval_metrics_generator(task_specs, hyperparams):
    """
    Returns the appropriate eval function depending on the task_specs.
    """
    return Metrics(task_specs.eval_metrics)


def compute_accuracy(output, target, prefix, topk=(1,)):
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
