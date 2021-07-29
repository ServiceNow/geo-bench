from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import pickle


class CustomEncoder(LightningModule):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(nn.Flatten(), nn.Linear(64 * 64 * 3, 512))

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":

    ce = CustomEncoder()
    torch.save(ce, "checkpoints/pt_model.pt")
