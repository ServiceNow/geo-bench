import torch


class ClassificationHead(torch.nn.Module):
    def __init__(self, in_ch, out_ch, hidden_size):
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_ch, hidden_size),
            torch.nn.Linear(hidden_size, out_ch)
            )

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        if len(x.size()) > 2:
            x = x.mean((2, 3))
        return self.linear(x)
