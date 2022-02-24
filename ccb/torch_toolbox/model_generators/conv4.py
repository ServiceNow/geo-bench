from typing import List
from ccb.experiment.experiment import hparams_to_string
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.model import BackBone, ModelGenerator, Model, head_generator, train_loss_generator
import torch.nn.functional as F
import torch


class Conv4Generator(ModelGenerator):
    def generate(self, task_specs: TaskSpecifications, hyperparameters: dict):
        """Returns a ccb.torch_toolbox.model.Model instance from task specs
           and hyperparameters

        Args:
            task_specs (TaskSpecifications): object with task specs
            hyperparameters (dict): dictionary containing hyperparameters
        """
        backbone = Conv4(self.model_path, task_specs, hyperparameters)
        head = head_generator(task_specs, hyperparameters)
        loss = train_loss_generator(task_specs, hyperparameters)
        return Model(backbone, head, loss, hyperparameters)

    def hp_search(self, task_specs, max_num_configs=10):
        hparams1 = {
            "lr_milestones": (10, 20),
            "lr_gamma": 0.1,
            "lr_backbone": 1e-3,
            "lr_head": 2e-3,
            "head_type": "linear",
            "train_iters": 50,
            "features_shape": (64,),
            "loss_type": "crossentropy",
            "batch_size": 64,
            "num_workers": 4,
            "logger": "csv",
        }

        hparams2 = hparams1.copy()
        hparams2["lr_head"] = 4e-3

        return hparams_to_string([hparams1, hparams2])


model_generator = Conv4Generator()


class Conv4(BackBone):
    def __init__(self, model_path, task_specs, hyperparams):
        super().__init__(model_path, task_specs, hyperparams)
        h, w, c, t = task_specs.patch_size
        self.conv0 = torch.nn.Conv2d(c, 64, 3, 1, 1)
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


# DATASETS = [
#     Dataset(
#         name="dataset1",
#         path="/dataset1/",
#         task_specs=TaskSpecifications(
#             input_shape=(1, 2, 3),
#             features_shape=(4, 5, 6),
#             spatial_resolution=10,
#             temporal_resolution=11,
#             band_names=["acdc", "queen"],
#             band_wavelength=0.2,
#             task_type="classification",
#             n_classes=10,
#         ),
#     ),
#     Dataset(
#         name="dataset2",
#         path="/dataset2/",
#         task_specs=TaskSpecifications(
#             input_shape=(1, 2, 3),
#             features_shape=(4, 5, 6),
#             spatial_resolution=2,
#             temporal_resolution=2,
#             band_names=["bob marley", "snoop dog"],
#             band_wavelength=0.1,
#             task_type="semantic segmentation",
#             n_classes=10,
#         ),
#     ),
# ]
