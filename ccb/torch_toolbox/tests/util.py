from typing import List
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.model import BackBone
import torch.nn.functional as F
import torch


class Conv4Example(BackBone):
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


# def iter_datasets():
#     """
#     Iterator over available datasets

#     """
#     for ds in DATASETS:
#         yield ds


# class Dataset(object):
#     def __init__(self, name: str, path: str, task_specs: List[TaskSpecifications]):
#         self.name = name
#         self.path = path
#         self.task_specs = task_specs


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