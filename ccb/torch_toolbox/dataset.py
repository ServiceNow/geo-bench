import torchvision
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from ccb.io import TaskSpecifications
from typing import List
import pytorch_lightning as pl


class Dataset(pl.LightningDataModule):
    def __init__(self, name: str, path: str, task_specs: TaskSpecifications, hyperparameters: dict):
        """Constructor. Downloads, splits, and provides dataloaders for a given dataset spec.

        Args:
            name (str): Dataset name. Ex. "eurosat"
            path (str): Path where the data is stored (or to be downloaded)
            task_specs (TaskSpecifications): Task specs that form this dataset
            hyperparameters (dict): Extra hyperparameters such as batch_size or num_workers.
        """
        # self.name = name
        self.path = path
        self.task_specs = task_specs
        self.hyperparameters = hyperparameters

    def prepare_data(self):
        if self.task_specs.dataset_name == "MNIST":
            torchvision.datasets.MNIST(self.path, train=True, download=True)
            torchvision.datasets.MNIST(self.path, train=False, download=True)

    def setup(self, stage=None):
        if self.task_specs.dataset_name == "MNIST":
            t = tt.ToTensor()
            self.train = torchvision.datasets.MNIST(self.path, train=True, transform=t, download=True)
            self.val = torchvision.datasets.MNIST(self.path, train=False, transform=t, download=True)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hyperparameters["batch_size"],
            shuffle=True,
            num_workers=self.hyperparameters["num_workers"],
        )

    def val_dataloader(self):
        train_batch_size = self.hyperparameters["batch_size"]  # default to train batch size if val not specified
        return DataLoader(
            self.val,
            batch_size=self.hyperparameters.get("val_batch_size", train_batch_size),
            shuffle=False,
            num_workers=self.hyperparameters["num_workers"],
        )
