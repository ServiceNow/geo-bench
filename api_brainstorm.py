
from my_module import MyBackBone, Classifier, SemanticSegmenter
import pl
import toolbox

backbone = MyBackBone(model_path="path_to_my_model")


#####
# fine_tuning.py
# TODO(pau, mehmet)
#######


class Model(pl.LightningModule):
    """
    Default Model class provided by the toolbox.

    TODO(pau)
    """
    def __init__(self, back_bone, head, loss_function, hyperparams):
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass



class TaskSpecifications:
    """
    Attributes:
        shape: (width, height, n_bands, time)
        spatial_resolution: physical distance between pixels in meters.
        temporal_resolution: Time difference between each frame in seconds. If time dimension > 1.
            (this feature might not make it into final version)
        band_names: Text describing each band in the dataset
        band_wavelenth: The central wavelength of each band.
        task_type: One of Classification, Semantic Segmentation, Counting ...
        n_classes: ...
        dataset_name: The name of the dataset.
    """
    def __init__(self, shape=None, spatial_resolution=None, temporal_resolution=None, band_names=None,
                 band_wevelength=None, task_type=None, dataset_name=None) -> None:
        self.shape = shape
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        self.band_names = band_names
        self.band_wevelength = band_wevelength
        self.task_type = task_type
        self.dataset_name = dataset_name

    def to_dict(self):
        return self.__dict__

    def from_dict(self, **kwargs):
        self.__dict__ = kwargs

####
# example_model_generator.py
# 
# Module defined by the user to specify how to wrap the pre-trained model and how to adapt it for each task, depending on task_specs.
# This module will be dynacmically loaded and our code will search for the variable model_generator, which sould implement hp_search and generate.
####

class MyBackBone:
    pass


class ModelGenerator:
    """
    Class implemented by the user. The goal is to specify how to connect the backbone with the head and the loss function.
    """
    def __init__(self, model_path) -> None:
        """This should not load the model at this point"""
        self.model_path = model_path

    def hp_search(self, task_specs, max_num_configs=10):
        """The user can provide a set of `max_num_configs` hyperparameters configuration to search for, based on task_specs"""
        return [dict(lr=0.4, width=100), dict(lr=0.1, width=100), dict(lr=0.1, width=200)]

    def generate(self, task_specs, hyperparams):
        backbone = MyBackBone(self.model_path, task_specs, hyperparams) # Implemented by the user so that he can wrap his 
        head = head_generator(task_specs, hyperparams) # provided by the toolbox or the user can implement his own
        loss = loss_generator(task_specs, hyperparams) # provided by the toolbox or the user can implement his own
        return Model(backbone, head, loss, hyperparams) # base model provided by the toolbox

model_geberator = ModelGenerator(model_path)                                           

####
# experiment_generator.py
####

#
# TODO(Dr. Ouin)
# 
# Script that takes as argument the user defined model generator e.g.: 
#   $ experiment_generator.py path/to/my/model/example_model_genartor.py
# 
# The model generator is loaded through dynamic import (to be discussed: is this good practice?)

# Example of directory structure: 
# 
# experiment-name_dd-mm-yy
# 	dataset-name1
# 		hp1=value1_hp2=value1_date=dd-mm-yy
# 			command_to_be_executed.bash
# 			job_specs.json
# 			result.json
# 			training_trace
# 			stdout
# 		hp1=value2_hp2=value1_date=dd-mm-yy
# 			...
# 		...
# dataset-name2
# 		hp1=value1_hp2=value1_date=dd-mm-yy
# 			...
# 		hp1=value2_hp2=value1_date=dd-mm-yy
# 		...

model_generator_path = argparser.model_generator_path
experiment_dir = argparser.experiment_dir


model_generator = dynamic_import(model_generator_path).model_genarator


def experiment_generator(model_generator, experiment_dir, task_filter=None, max_num_configs=10):
    """
    Generates the directory structure for every tasks and every hyperparameter configuration.
    According to model_generator.hp_search.
    """
    # TODO create experiment directory and append date in the dir name.
    for dataset in toolbox.iter_datasets():
        # TODO create directory for this dataset
        if task_filter is not None:
            if not task_filter(dataset.task_specs):
                continue

        for hyperparams in model_generator.hp_search(dataset.task_specs, max_num_configs):
            hyperparams_string = toolbox.hyperparams_to_string(hyperparams)
            # TODO 
            # * create directory with name reflecting hyperparameter configuration
            # * create short bash script to execute the job. File name should reflect hyperparams configuration 
            # * write hyperparams and task_specs in a json 
            pass

experiment_generator(model_generator, experiment_dir)

####
# trainer.py
# TODO(mehmet, pau)
#####

# script that dynamically load the user's model generator from arguments. Responsible for fine-tuning, validation,
# and writing results to the directory. 

model_generator_path = argparser.model_generator_path
model_generator = dynamic_import(model_generator_path).model_genarator

job_specs = json.load('job_specs.json') # job_specs.json is supposed to be in the current directory
task_specs = job_specs["task_specs"]
hyperparms = job_specs["hyperparams"]

train_loader, val_loader, test_loader = toolbox.data_loaders(task_specs.dataset_name)

model = model_generator(task_specs, hyperparms)

trainer = pl.Trainer()
trainer.fit(model, train_dataloaders=train_loader) # how to manage early stopping? can we pass the val_loader as well?

# TODO 
# * valid and test
# * write metrics in the results.json in the current directory
# * make sure some training statists are written to a training_trace 
#     this trace should be viewable in tensorboard or other tools such as weight an bias maybe
# * 
