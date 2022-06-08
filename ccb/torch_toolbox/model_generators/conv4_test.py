from ccb.experiment.experiment import hparams_to_string
from ccb.torch_toolbox.model_generators import conv4
from typing import Dict, Any


# class Conv4GeneratorTest(conv4.Conv4Generator):
#     def hp_search(self, task_specs, max_num_configs=10):
#         hparams1 = self.base_hparams.copy()
#         hparams1["train_iters"] = 10
#         hparams1["val_check_interval"] = 10
#         hparams1["num_workers"] = 0
#         hparams1["n_gpus"] = 0
#         hparams1["logger"] = "csv"
#         hparams1["optimizer"] = "adamW"
#         hparams2 = hparams1.copy()
#         hparams2["lr_head"] = 4e-3

#         return hparams_to_string([hparams1, hparams2])


def model_generator(hparams: Dict[str, Any] = {}) -> conv4.Conv4Generator:
    model_generator = conv4.Conv4Generator(hparams=hparams)
    return model_generator
