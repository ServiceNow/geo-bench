from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser
import warnings
warnings.simplefilter('ignore', UserWarning)

import torch
from torch import nn
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets.bigearthnet_datamodule import BigearthnetDataModule
from models.moco2_module import MocoV2
from models.ssl_finetuner import SSLFineTuner


def get_experiment_name(prefix, hparams):
    mode = 'linprobe' if hparams.freeze_backbone else 'finetune'
    return f'{prefix}-{mode}-lr={hparams.learning_rate}-epochs={hparams.max_epochs}-train_frac={hparams.train_frac}'


if __name__ == '__main__':
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = SSLFineTuner.add_model_specific_args(parser)
    parser = ArgumentParser(parents=[parser], conflict_handler='resolve', add_help=False)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--lmdb', action='store_true')
    parser.add_argument('--backbone_type', type=str, default='imagenet')
    parser.add_argument('--base_encoder', type=str, default='resnet18')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--train_frac', type=float, default=1)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    datamodule = BigearthnetDataModule(
        data_dir=args.data_dir,
        lmdb=args.lmdb,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_frac=args.train_frac
    )

    if args.backbone_type == 'random':
        template_model = getattr(models, args.base_encoder)
        backbone = template_model(pretrained=False)
        emb_dim = backbone.fc.weight.shape[1]
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        prefix = f'{args.base_encoder}-{args.backbone_type}'
    elif args.backbone_type == 'imagenet':
        template_model = getattr(models, args.base_encoder)
        backbone = template_model(pretrained=True)
        emb_dim = backbone.fc.weight.shape[1]
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        prefix = f'{args.base_encoder}-{args.backbone_type}'
    elif args.backbone_type == 'pretrain':
        model = MocoV2.load_from_checkpoint(args.ckpt_path)
        emb_dim = model.mlp_dim
        backbone = deepcopy(model.encoder_q)
        prefix = f'{model.hparams.base_encoder}-{args.backbone_type}-{model.hparams.data_mode}'
    else:
        raise ValueError()

    model = SSLFineTuner(
        backbone=backbone,
        in_features=emb_dim,
        num_classes=datamodule.num_classes,
        hidden_dim=None,
        **vars(args)
    )
    model.example_input_array = torch.zeros((1, 3, 128, 128))

    if args.debug:
        logger = False
        checkpoint_callback = False
    else:
        experiment_name = get_experiment_name(prefix, args)
        logger = TensorBoardLogger(save_dir=str(Path.cwd() / 'logs' / 'finetune'), name=experiment_name)
        checkpoint_callback = ModelCheckpoint(filename='{epoch}')

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        weights_summary='full',
        check_val_every_n_epoch=10
    )
    trainer.fit(model, datamodule=datamodule)
