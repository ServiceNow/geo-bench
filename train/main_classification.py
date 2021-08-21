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
from models.custom_encoder import BeforeLastLayerEncoder, FullModelEncoder, CLIPEncoder
from datasets.datamodule import DataModule


from models.moco2_module import MocoV2
from utils.utils import PretrainedModelDict, hp_to_str, get_arg_parser

import clip

# import onnx
# from onnx2pytorch import ConvertModel
# torch.autograd.detect_anomaly()


class Classifier(LightningModule):
    def __init__(self, in_features, num_classes, backbone=None):
        super().__init__()
        self.encoder = backbone
        self.classifier = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.f1 = F1()
        self.prec = Precision()
        self.rec = Recall()

    def forward(self, x):
        if self.encoder:
            x = self.encoder(x)
        x = x.float()
        logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx):

        loss, acc, f1, prec, rec = self.shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        self.log("train/f1", f1, prog_bar=True)
        self.log("train/prec", prec, prog_bar=True)
        self.log("train/rec", rec, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1, prec, rec = self.shared_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)
        self.log("val/prec", prec, prog_bar=True)
        self.log("val/rec", rec, prog_bar=True)

        return loss

    def shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(torch.argmax(logits, dim=1), y)
        f1 = self.f1(torch.argmax(logits, dim=1), y)
        prec = self.prec(torch.argmax(logits, dim=1), y)
        rec = self.rec(torch.argmax(logits, dim=1), y)

        return loss, acc, f1, prec, rec

    def configure_optimizers(self):

        max_epochs = self.trainer.max_epochs
        optimizer_params = [{"params": self.classifier.parameters(), "lr": args.lr}]

        if self.encoder:
            optimizer_params.append({"params": self.encoder.parameters(), "lr": args.backbone_lr})

        optimizer = optim.Adam(optimizer_params, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6 * max_epochs), int(0.8 * max_epochs)])

        return [optimizer], [scheduler]


if __name__ == "__main__":

    parser = get_arg_parser()
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    pmd = PretrainedModelDict()

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
    print(trainer.callback_metrics)

    with open(str(Path.cwd() / "logs" / experiment_name / "max_val"), "w") as f:
        f.write("max_accuracy: {}".format(torch.max(trainer.callback_metrics["val/acc"]).item()))
