from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser

import torch
from torch.nn import BCEWithLogitsLoss
from torchvision.models import resnet
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.classification import Precision, Recall, F1

from datasets.oscd_datamodule import ChangeDetectionDataModule
from models.segmentation import get_segmentation_model
from models.moco2_module import MocoV2


class SiamSegment(LightningModule):

    def __init__(self, backbone, feature_indices, feature_channels):
        super().__init__()
        self.model = get_segmentation_model(backbone, feature_indices, feature_channels)
        self.criterion = BCEWithLogitsLoss()
        self.prec = Precision(num_classes=1, threshold=0.5)
        self.rec = Recall(num_classes=1, threshold=0.5)
        self.f1 = F1(num_classes=1, threshold=0.5)

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1 = self.shared_step(batch)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/precision', prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/recall', rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        tensorboard.add_image('train/img_1', img_1[0], global_step)
        tensorboard.add_image('train/img_2', img_2[0], global_step)
        tensorboard.add_image('train/mask', mask[0], global_step)
        tensorboard.add_image('train/out', pred[0], global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1 = self.shared_step(batch)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/precision', prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recall', rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        tensorboard.add_image('val/img_1', img_1[0], global_step)
        tensorboard.add_image('val/img_2', img_2[0], global_step)
        tensorboard.add_image('val/mask', mask[0], global_step)
        tensorboard.add_image('val/out', pred[0], global_step)
        return loss

    def shared_step(self, batch):
        img_1, img_2, mask = batch
        out = self(img_1, img_2)
        pred = torch.sigmoid(out)
        loss = self.criterion(out, mask)
        prec = self.prec(pred, mask.long())
        rec = self.rec(pred, mask.long())
        f1 = self.f1(pred, mask.long())
        return img_1, img_2, mask, pred, loss, prec, rec, f1

    def configure_optimizers(self):
        # params = self.model.parameters()
        params = set(self.model.parameters()).difference(self.model.encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--backbone_type', type=str, default='imagenet')
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()

    datamodule = ChangeDetectionDataModule(args.data_dir)

    if args.backbone_type == 'random':
        backbone = resnet.resnet18(pretrained=False)
    elif args.backbone_type == 'imagenet':
        backbone = resnet.resnet18(pretrained=True)
    elif args.backbone_type == 'pretrain':
        model = MocoV2.load_from_checkpoint(args.ckpt_path)
        backbone = deepcopy(model.encoder_q)
    else:
        raise ValueError()

    model = SiamSegment(backbone, feature_indices=(0, 4, 5, 6, 7), feature_channels=(64, 64, 128, 256, 512))
    model.example_input_array = (torch.zeros((1, 3, 96, 96)), torch.zeros((1, 3, 96, 96)))

    experiment_name = args.backbone_type
    logger = TensorBoardLogger(save_dir=str(Path.cwd() / 'logs' / 'oscd'), name=experiment_name)
    checkpoint_callback = ModelCheckpoint(filename='{epoch}', save_weights_only=True)
    trainer = Trainer(gpus=args.gpus, logger=logger, callbacks=[checkpoint_callback], max_epochs=100, weights_summary='full')
    trainer.fit(model, datamodule=datamodule)
