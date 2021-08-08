from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser
import os
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
from utils.utils import hp_to_str


class SiamSegment(LightningModule):
    def __init__(self, backbone, feature_indices, feature_channels, finetune):
        super().__init__()
        self.model = get_segmentation_model(backbone, feature_indices, feature_channels)
        self.criterion = BCEWithLogitsLoss()
        self.prec = Precision(num_classes=1, threshold=0.5)
        self.rec = Recall(num_classes=1, threshold=0.5)
        self.f1 = F1(num_classes=1, threshold=0.5)
        self.finetune = finetune

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1 = self.shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/precision", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        tensorboard.add_image("train/img_1", img_1[0], global_step)
        tensorboard.add_image("train/img_2", img_2[0], global_step)
        tensorboard.add_image("train/mask", mask[0], global_step)
        tensorboard.add_image("train/out", pred[0], global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1 = self.shared_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/precision", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        tensorboard.add_image("val/img_1", img_1[0], global_step)
        tensorboard.add_image("val/img_2", img_2[0], global_step)
        tensorboard.add_image("val/mask", mask[0], global_step)
        tensorboard.add_image("val/out", pred[0], global_step)
        return loss

    def shared_step(self, batch):
        img_1, img_2, mask = batch
        out = self(img_1, img_2)
        pred = torch.sigmoid(out)
        loss = self.criterion(out, mask)

        print(torch.unique(mask.long()))
        prec = self.prec(pred.view(-1), mask.long().view(-1))
        rec = self.rec(pred.view(-1), mask.long().view(-1))
        f1 = self.f1(pred.view(-1), mask.long().view(-1))
        return img_1, img_2, mask, pred, loss, prec, rec, f1

    def configure_optimizers(self):

        optimizer_params = [
            {"params": list(set(self.model.parameters()).difference(self.model.encoder.parameters())), "lr": args.lr}
        ]

        if args.finetune:
            optimizer_params.append(
                {
                    "params": list(set(self.model.parameters()).difference(self.model.encoder.parameters())),
                    "lr": args.backbone_lr,
                }
            )

        optimizer = torch.optim.Adam(optimizer_params, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="datasets/oscd")
    parser.add_argument("--dataset", type=str, default="oscd")
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--backbone_type", type=str, default="imagenet")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--no_logs", action="store_false")

    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--backbone_lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    args = parser.parse_args()

    datamodule = ChangeDetectionDataModule(args)

    if args.backbone_type == "random":
        backbone = resnet.resnet18(pretrained=False)
    elif args.backbone_type == "imagenet":
        backbone = resnet.resnet18(pretrained=True)
    elif args.backbone_type == "pretrain":
        model = MocoV2.load_from_checkpoint(args.ckpt_path)
        backbone = deepcopy(model.encoder_q)
    elif args.backbone_type == "custom":
        backbone = torch.load(args.ckpt_path)
    else:
        raise ValueError('backbone_type must be one of "random", "imagenet", "custom" or "pretrain"')

    model = SiamSegment(
        backbone, feature_indices=(0, 4, 5, 6, 7), feature_channels=(64, 64, 128, 256, 512), finetune=args.finetune
    )
    model.example_input_array = (torch.zeros((1, 3, 96, 96)), torch.zeros((1, 3, 96, 96)))

    experiment_name = hp_to_str(args)

    os.makedirs(os.path.join(Path.cwd(), "logs", experiment_name), exist_ok=True)
    if args.no_logs:
        logger = TensorBoardLogger(save_dir=str(Path.cwd() / "logs"), name=experiment_name)
    else:
        logger = False

    checkpoint_callback = ModelCheckpoint(filename="{epoch}", save_weights_only=True)
    trainer = Trainer(
        gpus=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.max_epochs,
        weights_summary="full",
    )
    trainer.fit(model, datamodule=datamodule)

    with open(str(Path.cwd() / "logs" / experiment_name / "max_val"), "w") as f:
        max_idx = torch.argmax(trainer.callback_metrics["val/f1"])

        f.write(
            "max_f1_precision_recall: {} {} {}".format(
                trainer.callback_metrics["val/f1"].item(),
                trainer.callback_metrics["val/precision"].item(),
                trainer.callback_metrics["val/recall"].item(),
            )
        )
