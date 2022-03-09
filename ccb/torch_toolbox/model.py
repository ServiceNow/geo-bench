from typing import List
import torch
from pytorch_lightning import LightningModule
from ccb import io
import numpy as np


class Model(LightningModule):
    """
    Default Model class provided by the toolbox.

    TODO(pau-pow)
    """

    def __init__(self, backbone, head, loss_function, hyperparameters):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.loss_function = loss_function
        self.hyperparameters = hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss_train = self.loss_function(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log("train_loss", loss_train, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_acc5", acc5, on_step=True, on_epoch=True, logger=True)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix):
        images, target = batch
        output = self(images)
        loss = self.loss_function(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(f"{prefix}_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log(f"{prefix}_acc5", acc5, on_step=True, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

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

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


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
    if isinstance(task_specs.label_type, io.Classification):
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
    if isinstance(task_specs.label_type, io.Classification):
        if hyperparams["loss_type"] == "crossentropy":
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unrecognized loss type: {hyperparams['head_type']}")
    else:
        raise ValueError(f"Unrecognized task: {task_specs.label_type}")


def compute_metrics(output, target, topk=(1,)):
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
            res[f"accuracy-{k}"] = correct_k.mul_(100.0 / batch_size)
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
