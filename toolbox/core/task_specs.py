class TaskSpecifications:
    """
    Attributes:
        input_shape: (width, height, n_bands, time)
        features_shape: (channels, height, width)
        spatial_resolution: physical distance between pixels in meters.
        temporal_resolution: Time difference between each frame in seconds. If time dimension > 1.
            (this feature might not make it into final version)
        band_names: Text describing each band in the dataset
        band_wavelength: The central wavelength of each band.
        task_type: One of Classification, Semantic Segmentation, Counting ...
        n_classes: ...
        dataset_name: The name of the dataset.
        eval_loss: string specifying the type olf loss function used to evaluate the model on the validation set and the test set.
            (we should implement a dict mapping this string to loss_functions).
    """

    def __init__(
        self,
        input_shape=None,
        features_shape=None,
        spatial_resolution=None,
        temporal_resolution=None,
        band_names=None,
        band_wavelength=None,
        task_type=None,
        n_classes=None,
        dataset_name=None,
    ) -> None:
        self.input_shape = input_shape
        self.features_shape = features_shape
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        self.band_names = band_names
        self.band_wevelength = band_wavelength
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.n_classes = n_classes

    def to_dict(self):
        return self.__dict__

    def from_dict(self, **kwargs):
        self.__dict__ = kwargs

