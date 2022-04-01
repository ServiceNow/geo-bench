# from ccb.torch_toolbox.model_generators import conv4

from ccb.experiment.experiment_generator import experiment_generator
from toolkit import dispatch_toolkit

experiment_dir = experiment_generator(
    model_generator_module_name="ccb.torch_toolbox.model_generators.mae",
    experiment_dir="/mnt/data/experiments/huanggab",
    benchmark_name="imagenet",
    experiment_name="test_mae_imagenet",
)

dispatch_toolkit.push_code()
dispatch_toolkit.toolkit_dispatcher(experiment_dir)
