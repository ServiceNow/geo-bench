import torch
import torchvision
import torchvision.transforms as tt
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

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
        images, target = batch
        output = self(images)
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
        head_parameters = self.head.parameters()
        lr_backbone = self.hyperparameters['lr_backbone']
        lr_head = self.hyperparameters['lr_head']
        optimizer = torch.optim.Adam([{'params': backbone_parameters, 
                                        'lr': lr_backbone},
                                       {'params': head_parameters,
                                        'lr': lr_head}])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                        milestones=self.hyperparameters['lr_milestones'], 
                        gamma=self.hyperparameters['lr_gamma'])
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
    def __init__(self, model_path) -> None:
        """This should not load the model at this point"""
        self.model_path = model_path

    def hp_search(self, task_specs, max_num_configs=10):
        """The user can provide a set of `max_num_configs` hyperparameters configuration to search for, based on task_specs"""
        hp_configs = [dict(lr=0.4, width=100), dict(lr=0.1, width=100), dict(lr=0.1, width=200)]

        return hparams_to_string(hp_configs)

    def generate(self, task_specs, hyperparams):
        """Generate a Model to train

        Args:
            task_specs (TaskSpecifications): an object describing the task to be performed
            hyperparams (dict): dictionary containing hyperparameters of the experiment

        Raises:
            NotImplementedError

        Example:
            backbone = MyBackBone(self.model_path, task_specs, hyperparams) # Implemented by the user so that he can wrap his 
            head = head_generator(task_specs, hyperparams) # provided by the toolbox or the user can implement his own
            loss = train_loss_generator(task_specs, hyperparams) # provided by the toolbox or the user can implement his own
            return Model(backbone, head, loss, hyperparams) # base model provided by the toolbox
        """
        raise NotImplementedError()