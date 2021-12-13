import pytest
import torch
import torchvision
import torchvision.transforms as tt
from toolbox.model import Model
from toolbox import TaskSpecifications, head_generator, train_loss_generator
from toolbox.backbone import Conv4Example
import pytorch_lightning as pl


def test_toolbox_mnist():
    hyperparameters = {'lr_milestones': [10, 20],
                       'lr_gamma': 0.1,
                       'lr_backbone': 1e-3,
                       'lr_head': 1e-3,
                       'head_type': 'linear',
                       'train_iters': 100,
                       'loss_type': 'crossentropy'
                       }
    specs = TaskSpecifications((28, 28, 1, 1),
                               features_shape=(64,),
                               task_type='classification',
                               dataset_name='MNIST',
                               n_classes=10)

    head = head_generator(specs, hyperparameters)
    backbone = Conv4Example('./', specs, hyperparameters)
    criterion = train_loss_generator(specs, hyperparameters)
    model = Model(backbone,
                  head,
                  criterion,
                  hyperparameters=hyperparameters)

    t = tt.ToTensor()
    train_dataset = torchvision.datasets.MNIST('/mnt/public/datasets/mnist',
                                               transform=t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)

    trainer = pl.Trainer(gpus=0, max_epochs=1, max_steps=hyperparameters['train_iters'], logger=False)
    trainer.fit(model, train_dataloaders=train_loader)
    # print(trainer.logged_metrics)
    assert(trainer.logged_metrics['train_acc1_epoch'] > 11)  # has to be better than random after seeing 20 batches
