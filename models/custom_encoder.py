from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import pickle
from abc import ABC, abstractmethod


class BasicEncoder(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

    def forward(self, x):
        return self.backbone(x)


class FullModelEncoder(BasicEncoder):
    def __init__(self, model):
        super().__init__(model)
        self.backbone = model

    def forward(self, x):
        return self.backbone(x)


class CustomEncoder(LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(64 * 64 * 3, 1024), nn.Linear(1024, 512))
        self.net = nn.Linear(512, 10)

    def forward(self, x):
        return self.encoder(x)


class SegmentationEncoder(LightningModule):
    def __init__(self, backbone, feature_indices, diff=False):
        super().__init__()
        self.feature_indices = list(sorted(feature_indices))

        # A number of channels for each encoder feature tensor, list of integers
        # self._out_channels = feature_channels  # [3, 16, 64, 128, 256, 512]

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels = 3

        # Define encoder modules below
        self.encoder = backbone

        self.diff = diff

    def forward(self, x1, x2):
        """Produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
        """
        feats = [self.concatenate(x1, x2)]
        for i, module in enumerate(self.encoder.children()):
            x1 = module(x1)
            x2 = module(x2)
            if i in self.feature_indices:
                feats.append(self.concatenate(x1, x2))
            if i == self.feature_indices[-1]:
                break

        return feats

    def concatenate(self, x1, x2):
        if self.diff:
            return torch.abs(x1 - x2)
        else:
            torch.cat([x1, x2], 1)


if __name__ == "__main__":

    ce = CustomEncoder()
    torch.save(ce, "checkpoints/pt_model.pt")
