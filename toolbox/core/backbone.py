import torch
import torch.nn.functional as F


class MyBackBone(torch.nn.Module):
    def __init__(self, model_path, task_specs, hyperparams) -> None:
        super().__init__()
        self.model_path = model_path
        self.task_specs = task_specs
        self.hyperparams = hyperparams

    def forward(self, data_dict):
        """
        data_dict is a collection of tensors returned by the data loader.
        The user is responsible to implement something that will map
        the information from the dataset and encode it into a list of tensors.
        Returns: the encoded representation or a list of representations for
        models like u-net.
        raise NotImplementedError()
        """


class Conv4Example(MyBackBone):
    def __init__(self, model_path, task_specs, hyperparams):
        super().__init__(model_path, task_specs, hyperparams)
        h, w, c, t = task_specs.input_shape
        self.conv0 = torch.nn.Conv2d(c, 64, 3, 1, 1)
        self.conv1 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv0(x), True)
        x = F.relu(self.conv1(x), True)
        x = F.max_pool2d(x, 3, 2, 1)
        x = F.relu(self.conv2(x), True)
        x = F.max_pool2d(x, 3, 2, 1)
        x = F.relu(self.conv3(x), True)
        x = F.max_pool2d(x, 3, 2, 1)
        return x.mean((2, 3))
