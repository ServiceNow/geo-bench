from ccb.experiment.experiment import hparams_to_string
from ccb.torch_toolbox.model_generators import conv4


class Conv4GeneratorTest(conv4.Conv4Generator):
    def hp_search(self, task_specs, max_num_configs=10):
        hparams1 = self.base_hparams.copy()
        hparams1["train_iters"] = 50
        hparams1["n_gpus"] = 0

        hparams2 = hparams1.copy()
        hparams2["lr_head"] = 4e-3

        return hparams_to_string([hparams1, hparams2])


model_generator = Conv4GeneratorTest()
