import torch
import pytest
from ccb.torch_toolbox.modules import ClassificationHead


BATCH_SIZE = 2
NUM_CLASSES = 4
IN_CHANNEL = 5

@pytest.mark.parametrize("x", [torch.rand(size=(BATCH_SIZE, IN_CHANNEL)), [torch.rand(size=(BATCH_SIZE, IN_CHANNEL))], torch.rand(size=(BATCH_SIZE, IN_CHANNEL, IN_CHANNEL, IN_CHANNEL))])
def test_new_channel_init(x):

    cls_head = ClassificationHead(in_ch=IN_CHANNEL, num_classes=NUM_CLASSES, hidden_size=10)
    output = cls_head(x)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
