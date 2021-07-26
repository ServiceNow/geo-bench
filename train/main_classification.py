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
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

from datasets.eurosat_datamodule import EurosatDataModule
from datasets.sat_datamodule import SatDataModule
from models.moco2_module import MocoV2

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
print(parentdir)
sys.path.append(parentdir)


# from models.custom_encoder import CustomEncoder

# import onnx
# from onnx2pytorch import ConvertModel


sys.path.append("../datasets")


class Permute(torch.nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class Classifier(LightningModule):
    def __init__(self, in_features, num_classes, backbone=None):
        super().__init__()
        self.encoder = backbone
        self.classifier = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.max_val_accuracy = 0

    def forward(self, x):
        if self.encoder:

            x = self.encoder(x)

        logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx):

        loss, acc = self.shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

        self.max_val_accuracy = max(self.max_val_accuracy, acc.item())

        return loss

    def shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(torch.argmax(logits, dim=1), y)
        return loss, acc

    def configure_optimizers(self):

        max_epochs = self.trainer.max_epochs
        optimizer_params = [{"params": self.classifier.parameters(), "lr": args.lr}]

        if self.encoder:
            optimizer_params.append({"params": self.encoder.parameters(), "lr": args.bb_lr})

        optimizer = optim.Adam(optimizer_params)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6 * max_epochs), int(0.8 * max_epochs)])

        return [optimizer], [scheduler]


if __name__ == "__main__":
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--module", type=str)
    parser.add_argument("--class_name", type=str)

    parser.add_argument("--backbone_type", type=str, default="imagenet")
    parser.add_argument("--dataset", type=str, default="eurosat")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--exp", type=str, default="")
    parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bb_lr", type=float, default=0.001)
    parser.add_argument("--max_epochs", type=int, default=20)

    args = parser.parse_args()

    if args.backbone_type == "random":
        backbone = resnet.resnet18(pretrained=False)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    elif args.backbone_type == "imagenet":
        backbone = resnet.resnet18(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    elif args.backbone_type == "pretrain":
        model = MocoV2.load_from_checkpoint(args.ckpt_path)
        backbone = deepcopy(model.encoder_q)
    elif args.backbone_type == "dimport":
        module = __import__(args.module, fromlist=[args.class_name])
        model_class = getattr(module, args.class_name)
        backbone = model_class()
        backbone.load_state_dict(torch.load(args.ckpt_path))

        # with open(args.ckpt_path, 'rb') as handle:
        # backbone = pickle.load(handle)

    # elif args.backbone_type == 'onnx':
    #     model = onnx.load(args.ckpt_path)
    #     backbone = ConvertModel(model)#, experimental=True)# , debug=True
    #     backbone = nn.Sequential(Permute((0, 2, 3, 1)),backbone, nn.Flatten())

    else:
        raise ValueError('backbone_type must be one of "random", "imagenet", "custom" or "pretrain"')

    if args.dataset == "eurosat":
        datamodule = EurosatDataModule(args.data_dir)
    elif args.dataset == "sat":
        datamodule = SatDataModule(args.data_dir)
    else:
        raise ValueError('dataset must be one of "sat" or "eurosat"')

    if args.finetune:
        model = Classifier(in_features=512, num_classes=datamodule.num_classes, backbone=backbone)
        # model.example_input_array = torch.zeros((1, 3, 64, 64))

    else:
        datamodule.add_encoder(backbone)
        model = Classifier(in_features=512, num_classes=datamodule.num_classes)

    experiment_name = "{}-{}-{}-{}".format(args.dataset, args.lr, args.bb_lr, args.finetune)
    if args.exp:
        experiment_name = args.exp

    print(args.max_epochs)

    logger = TensorBoardLogger(save_dir=str(Path.cwd() / "logs"), name=experiment_name)
    trainer = Trainer(
        gpus=args.gpus, logger=logger, checkpoint_callback=False, max_epochs=args.max_epochs, weights_summary="full"
    )

    trainer.fit(model, datamodule=datamodule)

    with open(str(Path.cwd() / "logs" / experiment_name / "max_val_acc"), "w") as f:
        f.write(str(model.max_val_accuracy))
