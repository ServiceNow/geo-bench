# from ccb.torch_toolbox.model_generators import conv4

from ccb.experiment.experiment_generator import experiment_generator
from toolkit import dispatch_toolkit

expirment_dir = experiment_generator(
    model_generator_module_name="ccb.torch_toolbox.model_generators.conv4",
    experiment_dir="/mnt/data/experiments/allac",
    benchmark_name="classification",
    experiment_name="test_conv4",
)

dispatch_toolkit.push_code()
dispatch_toolkit.toolkit_dispatcher(expirment_dir)
