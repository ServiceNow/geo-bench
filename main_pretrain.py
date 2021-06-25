import os
from argparse import ArgumentParser
import warnings
warnings.simplefilter('ignore', UserWarning)

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.models.self_supervised.moco.callbacks import MocoLRScheduler

from datasets.seco_datamodule import SeasonalContrastBasicDataModule, SeasonalContrastTemporalDataModule, SeasonalContrastMultiAugDataModule
from models.moco2_module import MocoV2
from models.ssl_online import SSLOnlineEvaluator


def get_experiment_name(hparams):
    data_name = os.path.basename(hparams.data_dir)
    name = f'{hparams.base_encoder}-{data_name}-{hparams.data_mode}-epochs={hparams.max_epochs}'
    return name


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = MocoV2.add_model_specific_args(parser)
    parser = ArgumentParser(parents=[parser], conflict_handler='resolve', add_help=False)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--data_mode', type=str, choices=['moco', 'moco_tp', 'seco'], default='seco')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--schedule', type=int, nargs='*', default=[120, 160])
    parser.add_argument('--online_data_dir', type=str)
    parser.add_argument('--online_max_epochs', type=int, default=25)
    parser.add_argument('--online_val_every_n_epoch', type=int, default=25)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.data_mode == 'moco':
        datamodule = SeasonalContrastBasicDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    elif args.data_mode == 'moco_tp':
        datamodule = SeasonalContrastTemporalDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    elif args.data_mode == 'seco':
        datamodule = SeasonalContrastMultiAugDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        raise ValueError()

    model = MocoV2(**vars(args), emb_spaces=datamodule.num_keys)

    if args.debug:
        logger = False
        checkpoint_callback = False
    else:
        logger = TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), 'logs', 'pretrain'),
            name=get_experiment_name(args)
        )
        checkpoint_callback = ModelCheckpoint(filename='{epoch}')
    scheduler = MocoLRScheduler(initial_lr=args.learning_rate, schedule=args.schedule, max_epochs=args.max_epochs)
    online_evaluator = SSLOnlineEvaluator(
        data_dir=args.online_data_dir,
        z_dim=model.mlp_dim,
        max_epochs=args.online_max_epochs,
        check_val_every_n_epoch=args.online_val_every_n_epoch
    )

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[scheduler, online_evaluator],
        max_epochs=args.max_epochs,
        weights_summary='full'
    )
    trainer.fit(model, datamodule=datamodule)
