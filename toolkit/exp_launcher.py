"""Experiment Launcher with toolkit."""
# Convenient code for bypassing commandlines
# This will create experiments and lanch on toolkit
# This file serves as a template and will be remove from the repo once we're done with experiments.
# Please don't commit the changes related to your personnal experiments

import os
from pathlib import Path

from ccb.experiment.experiment_generator import experiment_generator
from toolkit import dispatch_toolkit

experiment_dir = experiment_generator(
    config_filepath=str(Path(__file__).parent.parent / "ccb/configs/classification_config.yaml"),
)

dispatch_toolkit.push_code(Path(__file__).parent.parent)

# you may want to change to your WANDB_API_KEY."
os.environ["WANDB_API_KEY"] = "def8d0fad10d1479d79ab4c7e68530d59be04cf5"
dispatch_toolkit.toolkit_dispatcher(experiment_dir, env_vars=(f"WANDB_API_KEY={os.environ['WANDB_API_KEY']}",))
