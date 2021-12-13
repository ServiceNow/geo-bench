
import pytorch_lightning as pl
import toolbox


#####
# toolbox.py
# Can be split across various files
#
# TODO add data loaders.
#######


class Model(pl.LightningModule):
    """
    Default Model class provided by the toolbox.

    TODO(pau-pow)
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
        eval_loss: string specifying the type olf loss function used to evaluate the model on the validation set and the test set.
            (we should implement a dict mapping this string to loss_functions).
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


def head_generator(task_specs, hyperparams, input_shape):
    """
    Returns a an appropriate head based on the task specifications. We can use task_specs.task_type as follow: 
        classification: 2 layer MLP with softmax activation
        semantic_segmentation: U-Net decoder. 
    we can also do something special for a specific dataset using task_specs.dataset_name. Hyperparams and input_shape 
    can also be used to adapt the head.

    Args:
        task_specs: object of type TaskSpecifications providing information on what type of task we are solving
        hyperparams: dict of hyperparameters.
        input_shape: list of tuples describing the shape of the input of this module. TO BE DISCUSSED: should this be
            the input itself? should it be a dict of shapes? 
    """
    if task_specs.task_type.lower() == "classification":
        return LinearHead(input_shape[-1], output_size=task_specs.n_classes)
    elif task_specs.task_type.lower() == "segmentation":
        pass  # TODO do something for segmentation, but later.
    else:
        raise NotImplemented(f"Unknown task type {task_specs.task_type}.")


class LinearHead(pl.LightningModule):

    def __init__(self, input_size, output_size):
        super.__init__()
        self._build(input_size, output_size)

    def _build(sefl, input_size, output_size):
        pass

    def forward(self, input):
        pass


def vit_head_generator(task_specs, hyperparams, input_shape):
    """
    ViT architectures may require different type of heads. In which case, we should provide this to the users as well. TO BE DISCUSSED. 
    """
    pass


def train_loss_generator(task_specs, hyperparams):
    """
    Returns the appropriate loss function depending on the task_specs. We should implement basic loss and we can leverage the
    following attributes: task_specs.task_type and task_specs.eval_loss
    """


def hparams_to_string(list_of_hp_configs):
    """
    Generate a string respresentation of the meaningful hyperparameters. This string will be used for file names and job names, to be able
    to distinguish them easily.

    TODO: Introspect the list of hyperparms configurations to find the hyperparameters that changes during the experiment e.g.,
    there might be 8 hyperparameters but only 2 that changes. Format the changing hyperparamters into a short string that can be used for filenames. 
    """

####
# Userside:
# example_model_generator.py
#
# Module defined by the user to specify how to wrap the pre-trained model and how to adapt it for each task, depending on task_specs.
# This module will be dynacmically loaded and our code will search for the variable model_generator, which sould implement hp_search and generate.
####


class MyBackBone:
    def __init__(self, model_path, task_specs, hyperparams) -> None:
        self.model_path = model_path
        self.task_specs = task_specs
        self.hyperparams = hyperparams

    def forward(self, data_dict):
        # data_dict is a collection of tensors returned by the data loader.
        # The user is responsible to implement something that will map
        # the information from the dataset and encode it into a list of tensors.
        # Returns: the encoded representation or a list of representations for
        #    models like u-net.
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
        hp_configs = [dict(lr=0.4, width=100), dict(lr=0.1, width=100), dict(lr=0.1, width=200)]

        return hparams_to_string(hp_configs)

    def generate(self, task_specs, hyperparams):
        # Implemented by the user so that he can wrap his
        backbone = MyBackBone(self.model_path, task_specs, hyperparams)
        head = head_generator(task_specs, hyperparams)  # provided by the toolbox or the user can implement his own
        # provided by the toolbox or the user can implement his own
        loss = train_loss_generator(task_specs, hyperparams)
        return Model(backbone, head, loss, hyperparams)  # base model provided by the toolbox


model_generator = ModelGenerator(model_path)

####
# experiment_generator.py
####
#
# TODO(Dr. Ouin)
# * materilaze this pseudocode
# * implement experiment_generator_test.py, which would generate a fake structure in /tmp and verify the
#     content of it. (no need to verify every details, but a quick checkup)
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

        for hyperparams, hyperparams_string in model_generator.hp_search(dataset.task_specs, max_num_configs):
            # TODO
            # * create directory with name reflecting hyperparameter configuration using hyperparams_string
            # * generate a short bash script to execute the job. File name should contain hyperparams_string
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

job_specs = json.load('job_specs.json')  # job_specs.json is supposed to be in the current directory
task_specs = job_specs["task_specs"]
hyperparms = job_specs["hyperparams"]

train_loader, val_loader, test_loader = toolbox.data_loaders(task_specs.dataset_name)

model = model_generator.generate(task_specs, hyperparms)

trainer = pl.Trainer()
trainer.fit(model, train_dataloaders=train_loader)  # how to manage early stopping? can we pass the val_loader as well?

# TODO
# * valid and test
# * write metrics in the results.json in the current directory
# * make sure some training statists are written to a training_trace
#     this trace should be viewable in tensorboard or other tools such as weight an bias maybe
# *


######
#  Integration test
# TODO(mehmet)
#####
# * make a very small convenet backbone with random init.
# * wrap it in a Model with a classifier head to mockup a user implementation.
# * generate experiments with 2 hyperparam configurations and 1 dataset: MNIST
# * Execute all experiments with a simple script that execute sequentially on local machine
#       * Train for e.g. 10 steps or until there is at least 2-3 points in the training_trace.
#       * Run the eval procedure and write all result files.
# * preform a few sanity checks to make sure that all files that should be there are there
#       and that they are readable and contains the expected information.
#
# Hopefully this test could run in less than a minute without the need of GPU. If not maybe we can
# generate a dataset even smaller and configure the writing to training_traces every step.
