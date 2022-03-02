import torchvision
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from ccb import io
from typing import List
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(
        self, task_specs: io.TaskSpecifications, batch_size: int, num_workers: int, val_batch_size: int = None
    ):
        """DataModule providing dataloaders from task_specs.

        Args:
            task_specs: TaskSpecifications object to call get_dataset.
            batch_size: The size of the mini-batch.
            num_workers: The number of parallel workers for loading samples from the hard-drive.
            val_batch_size: Tes size of the batch for the validation set and test set. If None, will use batch_size.
        """
        self.task_specs = task_specs
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.task_specs.get_dataset(split="train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.task_specs.get_dataset(split="valid"),
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
