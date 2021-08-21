from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser
import os
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torchvision.models import resnet
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# from pytorch_lightning.metrics.classification import Precision, Recall, F1, IoU
from torchmetrics import Precision, Recall, F1, IoU

from models.moco2_module import MocoV2
from utils.utils import hp_to_str, get_arg_parser
from models.custom_encoder import SegmentationEncoder
from models.segmentation import UNet
from datasets.datamodule import DataModule


class SiamSegment(LightningModule):
    def __init__(self, backbone, feature_channels, finetune, num_classes):
        super().__init__()

        self.model = UNet(backbone, feature_channels, num_classes, bilinear=True, concat_mult=1, dropout_rate=0.3)

        if num_classes == 1:
            self.criterion = BCEWithLogitsLoss()
        else:
            self.criterion = CrossEntropyLoss()

        self.iou = IoU(num_classes=num_classes if num_classes != 1 else 2, threshold=0.5, reduction="elementwise_mean")

        if num_classes == 1:
            self.f1_macro = None
            num_classes = None
        else:
            self.f1_macro = F1(num_classes=num_classes, threshold=0.5, average="macro", mdmc_average="global")

        self.prec = Precision(num_classes=num_classes, threshold=0.5, mdmc_average="global")
        self.rec = Recall(num_classes=num_classes, threshold=0.5, mdmc_average="global")
        self.f1_micro = F1(num_classes=num_classes, threshold=0.5, average="micro", mdmc_average="global")

        self.finetune = finetune

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1_micro, f1_macro, iou = self.shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)

        # each sample is weighed equally, regardless of class
        self.log("train/precision", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/recall",
            rec,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train/f1_micro", f1_micro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        if self.f1_macro:
            self.log("train/f1_macro", f1_macro, on_step=False, on_epoch=True, prog_bar=True)

        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step

        if mask.ndim < 4:
            mask = mask.unsqueeze(1)
        tensorboard.add_image("train/img_1", img_1[0], global_step)
        tensorboard.add_image("train/img_2", img_2[0], global_step)
        tensorboard.add_image("train/mask", mask[0], global_step)
        tensorboard.add_image("train/out", pred[0, :3], global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1_micro, f1_macro, iou = self.shared_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/precision", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1_micro", f1_micro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        if self.f1_macro:
            self.log("val/f1_macro", f1_macro, on_step=False, on_epoch=True, prog_bar=True)

        if mask.ndim < 4:
            mask = mask.unsqueeze(1)
        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        tensorboard.add_image("val/img_1", img_1[0], global_step)
        tensorboard.add_image("val/img_2", img_2[0], global_step)
        tensorboard.add_image("val/mask", mask[0], global_step)
        tensorboard.add_image("val/out", pred[0, :3], global_step)  # TODO: add each channel?

        return loss

    def shared_step(self, batch):
        img_1, img_2, mask = batch

        out = self(img_1, img_2)
        pred = torch.sigmoid(out)

        loss = self.criterion(out, mask)

        mask = mask.long()

        prec = self.prec(pred, mask)
        rec = self.rec(pred, mask)
        f1_micro = self.f1_micro(pred, mask.long())
        iou = self.iou(pred, mask.long())

        if self.f1_macro:
            f1_macro = self.f1_macro(pred, mask.long())
        else:
            f1_macro = None

        return img_1, img_2, mask, pred, loss, prec, rec, f1_micro, f1_macro, iou

    def configure_optimizers(self):

        optimizer_params = [
            {"params": list(set(self.model.parameters()).difference(self.model.encoder.parameters())), "lr": args.lr}
        ]

        if args.finetune:
            optimizer_params.append(
                {
                    "params": self.model.encoder.parameters(),
                    "lr": args.backbone_lr,
                }
            )

        optimizer = torch.optim.Adam(optimizer_params, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]


if __name__ == "__main__":

    parser = get_arg_parser()
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    datamodule = DataModule(args)

    if args.backbone_type == "random":
        backbone = resnet.resnet18(pretrained=False)
        backbone = SegmentationEncoder(backbone, feature_indices=(0, 4, 5, 6, 7), diff=True)

    elif args.backbone_type == "imagenet":
        backbone = resnet.resnet18(pretrained=True)
        backbone = SegmentationEncoder(backbone, feature_indices=(0, 4, 5, 6, 7), diff=True)

    elif args.backbone_type == "seco":
        model = MocoV2.load_from_checkpoint(args.ckpt_path)
        backbone = deepcopy(model.encoder_q)
        backbone = SegmentationEncoder(backbone, feature_indices=(0, 4, 5, 6, 7), diff=True)

    elif args.backbone_type == "custom":
        backbone = torch.load(args.ckpt_path)

    else:
        raise ValueError('backbone_type must be one of "random", "imagenet", "custom" or "seco"')

    model = SiamSegment(
        backbone,
        feature_channels=(64, 64, 128, 256, 512),
        finetune=args.finetune,
        num_classes=datamodule.get_num_classes(),
    )

    model.example_input_array = (torch.zeros((1, 3, 96, 96)), torch.zeros((1, 3, 96, 96)))

    experiment_name = hp_to_str(args)

    os.makedirs(os.path.join(Path.cwd(), "logs", experiment_name), exist_ok=True)

    if args.no_logs:
        logger = False
    else:
        logger = TensorBoardLogger(save_dir=str(Path.cwd() / "logs"), name=experiment_name)

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
