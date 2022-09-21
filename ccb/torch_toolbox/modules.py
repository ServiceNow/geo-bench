"""Modules."""

from typing import Any, List, Union

import torch
from torch import Tensor


class ClassificationHead(torch.nn.Module):
    """Classification Head.

    Define a two layer classification head that can be attached to model backbone.
    """

    def __init__(self, in_ch: int, num_classes: int, ret_identity: bool = False) -> None:
        """Initialize new instance of Classification Head.

        Args:
            in_ch: number of input channels
            num_classes: number of classes to predict
            ret_identy: whether or not just return the feature input
                (for example smp models have their own classification head)
        """
        super().__init__()
        self.num_classes = num_classes

        self.linear = torch.nn.Sequential(torch.nn.Linear(in_ch, num_classes))

        self.ret_identity = ret_identity

    def forward(self, x: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
        """Forward input through classification head.

        Args:
            x: input

        Returns:
            model predictions
        """
        if self.ret_identity:
            return x
        else:
            if isinstance(x, list):
                x = x[-1]
            if len(x.size()) > 2:
                x = x.mean((2, 3))
            return self.linear(x)
