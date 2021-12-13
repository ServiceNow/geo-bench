import torch

class TaskSpecifications:
    """
    Attributes:
        input_shape: (width, height, n_bands, time)
        features_shape: (channels, height, width)
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
    def __init__(self, input_shape=None, features_shape=None, spatial_resolution=None, 
                    temporal_resolution=None, band_names=None, band_wevelength=None, 
                    task_type=None, n_classes=None, dataset_name=None) -> None:
        self.input_shape = input_shape
        self.features_shape = features_shape
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        self.band_names = band_names
        self.band_wevelength = band_wevelength
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.n_classes = n_classes

    def to_dict(self):
        return self.__dict__

    def from_dict(self, **kwargs):
        self.__dict__ = kwargs


def head_generator(task_specs, hyperparams):
    """
    Returns a an appropriate head based on the task specifications. We can use task_specs.task_type as follow: 
        classification: 2 layer MLP with softmax activation
        semantic_segmentation: U-Net decoder. 
    we can also do something special for a specific dataet using task_specs.dataset_name. Hyperparams and input_shape 
    can also be used to adapt the head.

    Args:
        task_specs: object of type TaskSpecifications providing information on what type of task we are solving
        hyperparams: dict of hyperparameters.
        input_shape: list of tuples describing the shape of the input of this module. TO BE DISCUSSED: should this be
            the input itself? should it be a dict of shapes? 
    """
    if task_specs.task_type == 'classification':
        if hyperparams['head_type'] == 'linear':
            in_ch, = task_specs.features_shape
            out_ch = task_specs.n_classes
            return torch.nn.Linear(in_ch, out_ch)
        else:
            raise ValueError(f"Unrecognized head type: {hyperparams['head_type']}")
    else:
        raise ValueError(f"Unrecognized task: {task_specs.task_type}")



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
    Introspect the list of hyperparms configurations to find the hyperparameters that changes during the experiment e.g.,
    there might be 8 hyperparameters but only 2 that changes. For each hyperparameter that changes 
    """