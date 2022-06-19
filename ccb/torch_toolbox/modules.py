"""Modules."""

from typing import Any

import torch
from torch import Tensor


class ClassificationHead(torch.nn.Module):
    """Classification Head.

    Define a two layer classification head that can be attached to model backbone.
    """

    def __init__(self, in_ch: int, num_classes: int, hidden_size: int) -> None:
        """Initialize new instance of Classification Head.

        Args:
            in_ch: number of input channels
            num_classes: number of classes to predict
            hidden_size: hidden size of linear layer
        """
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_ch, hidden_size), torch.nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: Any) -> Tensor:
        """Forward input through classification head.

        Args:
            x: input

        Returns:
            model predictions
        """
        if isinstance(x, list):
            x = x[-1]
        if len(x.size()) > 2:
            x = x.mean((2, 3))
        return self.linear(x)
