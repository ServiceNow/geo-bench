import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models
from argparse import ArgumentParser
from torchvision.transforms import functional as TF
import random


def RandomFlip(*xs):
    if random.random() > 0.5:
        xs = tuple(TF.hflip(x) for x in xs)
    return xs


def RandomRotation(*xs):
    angle = random.choice([0, 90, 180, 270])
    return tuple(TF.rotate(x, angle) for x in xs)


def RandomSwap(x1, x2, y):
    if random.random() > 0.5:
        return x2, x1, y
    else:
        return x1, x2, y


def ToTensor(*xs):
    return tuple(TF.to_tensor(x) for x in xs)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *xs):
        for t in self.transforms:
            xs = t(*xs)
        return xs


def get_embeddings(encoder, dataset, bs=128):
    embeddings = None
    dl = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=8, drop_last=False)

    encoder = encoder.cuda().eval()

    with torch.no_grad():
        for i, (batch, _) in tqdm(enumerate(dl)):
            encoder = encoder.cuda()
            batch_embs = encoder(batch.cuda())

            if embeddings is None:
                embeddings = torch.zeros((len(dataset), batch_embs.shape[-1]))
            embeddings[i * bs : i * bs + batch_embs.shape[0]] = batch_embs

    dataset.set_embeddings(embeddings.float())


def hp_to_str(args):
    return "{}-{}-{}-{}-{}-{}".format(
        args.dataset, args.backbone_type, args.lr, args.backbone_lr, args.weight_decay, args.finetune
    )


def get_arg_parser():

    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="datasets/eurosat")
    parser.add_argument("--module", type=str)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_logs", action="store_true")

    parser.add_argument("--backbone_type", type=str, default="imagenet")
    parser.add_argument("--dataset", type=str, default="eurosat")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--feature_size", type=int, default=512)

    parser.add_argument(
        "--batch_size", type=int, default=128, help="Baseline: 128 for clasification, 32 for segmentation"
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--backbone_lr", type=float, default=0.001)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--patch_size", type=int, default=96)

    return parser


class PretrainedModelDict:
    def __init__(self):
        self.model_dict = {
            "resnet18": models.resnet18,
            "alexnet": models.alexnet,
            "vgg16": models.vgg16,
            "squeezenet": models.squeezenet1_0,
            "densenet": models.densenet161,
            "inception": models.inception_v3,
            "googlenet": models.googlenet,
            "shufflenet": models.shufflenet_v2_x1_0,
            "mobilenet_v2": models.mobilenet_v2,
            "mobilenet_v3_large": models.mobilenet_v3_large,
            "mobilenet_v3_small": models.mobilenet_v3_small,
            "resnext50_32x4d": models.resnext50_32x4d,
            "wide_resnet50_2": models.wide_resnet50_2,
            "mnasnet": models.mnasnet1_0,
        }

    def get_model(self, model):
        return self.model_dict[model](pretrained=True)

    def get_available_models(self):
        return self.model_dict.keys()
