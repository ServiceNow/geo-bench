import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models


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
