# Convenient code for bypassing commandlines
# This file serves as a template and will be remove from the repo once we're done with experiments.
# Please don't commit the changes related to your personnal experiments

from pathlib import Path
from ccb.experiment.experiment_generator import experiment_generator
from toolkit import dispatch_toolkit

ccb_code_dir = Path(__file__).parent.parent


expirment_dir = experiment_generator(
    model_generator_module_name="ccb.torch_toolbox.model_generators.conv4",
    experiment_dir="/mnt/data/experiments/allac",
    benchmark_name="classification",
    experiment_name="test_conv4",
)


dispatch_toolkit.push_code(ccb_code_dir)
dispatch_toolkit.toolkit_dispatcher(expirment_dir)