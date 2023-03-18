"""GeobenchDataset Datamodule."""

from pathlib import Path
from typing import Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from geobench import io


class DataModule(pl.LightningDataModule):
    """Data Module.

    Define a
    `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html>`_
    that provides dataloaders from task_specs.
    """

    def __init__(
        self,
        task_specs: io.TaskSpecifications,
        benchmark_dir: str,
        partition_name: str,
        batch_size: int,
        num_workers: int,
        val_batch_size: int = None,
        train_transform=None,
        eval_transform=None,
        collate_fn=None,
        band_names: Sequence[str] = ("red", "green", "blue"),
        format: str = "hdf5",
    ) -> None:
        """Initialize new instance of DataModule .

        Args:
            task_specs: TaskSpecifications object to call get_dataset.
            benchmark_dir: path to benchmark directory that contains datasets
            partition_name: name of partition to load
            batch_size: The size of the mini-batch.
            num_workers: The number of parallel workers for loading samples from the hard-drive.
            val_batch_size: Tes size of the batch for the validation set and test set. If None, will use batch_size.
            transform: Callable transforming a Sample. Executed on a worker and the output will be provided to collate_fn.
            collate_fn: A callable passed to the DataLoader. Maps a list of Sample to dictionnary of stacked torch tensors.
            band_names: multi spectral bands to select
            file_format: 'hdf5' or 'tif'
        """
        super().__init__()
        self.task_specs = task_specs
        self.benchmark_dir = benchmark_dir
        self.partition_name = partition_name
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.collate_fn = collate_fn
        self.band_names = band_names
        self.format = format

    def train_dataloader(self) -> DataLoader:
        """Create the train dataloader."""
        return DataLoader(
            self.task_specs.get_dataset(
                split="train",
                partition_name=self.partition_name,
                transform=self.train_transform,
                band_names=self.band_names,
                format=self.format,
                benchmark_dir=self.benchmark_dir,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader."""
        return (
            DataLoader(
                self.task_specs.get_dataset(
                    split="valid",
                    partition_name=self.partition_name,
                    transform=self.eval_transform,
                    band_names=self.band_names,
                    format=self.format,
                    benchmark_dir=Path(self.benchmark_dir),
                ),
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            ),
            DataLoader(
                self.task_specs.get_dataset(
                    split="test",
                    partition_name=self.partition_name,
                    transform=self.eval_transform,
                    band_names=self.band_names,
                    format=self.format,
                    benchmark_dir=Path(self.benchmark_dir),
                ),
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            ),
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test dataloader."""
        return DataLoader(
            self.task_specs.get_dataset(
                split="test",
                partition_name=self.partition_name,
                transform=self.eval_transform,
                band_names=self.band_names,
                format=self.format,
                benchmark_dir=self.benchmark_dir,
            ),
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
