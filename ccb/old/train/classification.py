from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser
import os
import sys
import pickle
import torch
from torch import nn, optim
from torchvision.models import resnet
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer

# from pytorch_lightning.metrics import Accuracy
from torchmetrics import Precision, Recall, F1, Accuracy

from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from ccb.models.custom_encoder import BeforeLastLayerEncoder, FullModelEncoder, CLIPEncoder
from ccb.datasets.datamodule import DataModule


from ccb.models.moco2_module import MocoV2
from ccb.utils.utils import PretrainedModelDict, hp_to_str, get_arg_parser

import clip
import sklearn.metrics

# import onnx
# from onnx2pytorch import ConvertModel
# torch.autograd.detect_anomaly()


class Classifier(LightningModule):
    def __init__(self, in_features, num_classes, backbone=None, trainer=None, args=None):
        super().__init__()
        self.args = args
        self.encoder = backbone
        self.trainer = trainer
        self.classifier = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(average="micro")
        self.f1 = F1(num_classes=num_classes, average="macro")
        self.prec = Precision(num_classes=num_classes, average="macro")
        self.rec = Recall(num_classes=num_classes, average="macro")

        # self.targets = torch.tensor([]).cuda()
        # self.preds = torch.tensor([]).cuda()

        self.best_epoch_metrics = None

    def forward(self, x):
        if self.encoder:
            x = self.encoder(x)
        x = x.float()
        logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx):

        loss, acc, f1, prec, rec, _, _ = self.shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        self.log("train/f1", f1, prog_bar=True)
        self.log("train/prec", prec, prog_bar=True)
        self.log("train/rec", rec, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1, prec, rec, preds, targets = self.shared_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)
        self.log("val/prec", prec, prog_bar=True)
        self.log("val/rec", rec, prog_bar=True)

        # for i in range(acc.shape[0]):
        # self.log("val/acc" + str(i), acc[i], prog_bar=True)

        # self.targets = torch.cat((targets, self.targets))
        # self.preds = torch.cat((preds, self.preds))

        return loss

    def shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1(preds, y)
        prec = self.prec(preds, y)
        rec = self.rec(preds, y)

        return loss, acc, f1, prec, rec, preds, y

    def validation_epoch_end(self, validation_step_outputs):

        if self.best_epoch_metrics is None:
            self.best_epoch_metrics = self.trainer.logged_metrics

        if self.trainer.logged_metrics["val/loss"] < self.best_epoch_metrics["val/loss"]:
            self.best_epoch_metrics = self.trainer.logged_metrics
        # print("=================\n", self.best_epoch_metrics, "\n-------------------------------\n", trainer.logged_metrics, "==========================\n")

        # self.targets = torch.tensor([]).cuda()
        # self.preds = torch.tensor([]).cuda()

    def configure_optimizers(self):
        max_epochs = self.trainer.max_epochs
        optimizer_params = [{"params": self.classifier.parameters(), "lr": self.args.lr}]

        if self.encoder:
            optimizer_params.append({"params": self.encoder.parameters(), "lr": self.args.backbone_lr})

        optimizer = optim.Adam(optimizer_params, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6 * max_epochs), int(0.8 * max_epochs)])

        return [optimizer], [scheduler]


def start():
    parser = get_arg_parser()
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # pmd = PretrainedModelDict()

    if args.backbone_type == "random":
        backbone = BeforeLastLayerEncoder(resnet.resnet18(pretrained=False))
    elif args.backbone_type == "imagenet":
        backbone = BeforeLastLayerEncoder(resnet.resnet18(pretrained=True))
    elif args.backbone_type == "seco":  # to load seco
        model = MocoV2.load_from_checkpoint(args.ckpt_path)
        backbone = FullModelEncoder(deepcopy(model.encoder_q))
    elif args.backbone_type == "custom":
        backbone = FullModelEncoder(torch.load(args.ckpt_path))

    # elif args.backbone_type in pmd.get_available_models(): # only tested resnet18 for now
    #     backbone = pmd.get_model(args.backbone_type)

    #     # print(list(backbone.children()))
    #     backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())

    elif args.backbone_type in clip.available_models():  # currently get nan losses
        # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
        model, preprocess = clip.load(args.backbone_type, jit=False)
        backbone = CLIPEncoder(model.float(), preprocess)

    # elif args.backbone_type == 'onnx':
    #     model = onnx.load(args.ckpt_path)
    #     backbone = ConvertModel(model)#, experimental=True)# , debug=True
    #     backbone = nn.Sequential(Permute((0, 2, 3, 1)),backbone, nn.Flatten())

    else:
        raise ValueError('backbone_type must be one of "random", "imagenet", "custom" or "seco"')

    datamodule = DataModule(args)

    if args.finetune:
        model = Classifier(in_features=args.feature_size, num_classes=datamodule.get_num_classes(), backbone=backbone)
        # model.example_input_array = torch.zeros((1, 3, 64, 64))

    else:
        datamodule.add_encoder(backbone)
        model = Classifier(in_features=args.feature_size, num_classes=datamodule.get_num_classes())

    experiment_name = hp_to_str(args)

    os.makedirs(os.path.join(Path.cwd(), "logs", experiment_name), exist_ok=True)

    if args.no_logs:
        logger = False
    else:
        logger = TensorBoardLogger(save_dir=str(Path.cwd() / "logs"), name=experiment_name)

    trainer = Trainer(
        gpus=args.gpus,
        logger=logger,
        checkpoint_callback=False,
        max_epochs=args.max_epochs,
        weights_summary="full",
        terminate_on_nan=True,
    )

    trainer.fit(model, datamodule=datamodule)

    # min_loss_idx = torch.argmin(trainer.callback_metrics["val/loss"])

    with open(str(Path.cwd() / "logs" / experiment_name / "max_val"), "w") as f:
        # f.write(
        #     "max_accuracy_f1: {} {}".format(
        #         torch.max(trainer.callback_metrics["val/acc0"]).item(),
        #         torch.max(trainer.callback_metrics["val/f1"]).item(),
        #     )
        # )
        f.write("epoch,acc,f1,rec,prec\n")
        f.write(
            "{},{},{},{},{}".format(
                model.best_epoch_metrics["epoch"],
                model.best_epoch_metrics["val/acc"],
                model.best_epoch_metrics["val/f1"],
                model.best_epoch_metrics["val/rec"],
                model.best_epoch_metrics["val/prec"],
            )
        )
