from ccb.torch_toolbox.model_generators import conv4
from typing import Dict, Any


def model_generator(hparams: Dict[str, Any] = {}) -> conv4.Conv4Generator:
    model_generator = conv4.Conv4Generator(hparams=hparams)
    return model_generator
