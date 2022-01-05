from copy import deepcopy
import pickle
from ccb.dataset import io
from pathlib import Path


class TaskSpecifications:
    """
    Attributes:
        dataset_name: The name of the dataset.
        patch_size: maximum image patch size across bands (width, height).
        spatial_resolution: physical distance between pixels in meters.
        bands_info: list of object of type BandInfo descrbing the type of each band
        label_type: One of Classification, Semantic Segmentation, Counting ...
        eval_loss: string specifying the type of loss function used to evaluate the model on the validation set and the test set.
            (we should implement a dict mapping this string to loss_functions).
    """

    def __init__(
        self,
        dataset_name=None,
        patch_size=None,
        n_time_steps=None,
        bands_info=None,
        bands_stats=None,
        label_type=None,
        eval_loss=None,
        spatial_resolution=None,
    ) -> None:
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.n_time_steps = n_time_steps
        self.bands_info = bands_info
        self.bands_stats = bands_stats
        self.label_type = label_type
        self.eval_loss = eval_loss
        self.spatial_resolution = spatial_resolution

    def save(self, directory):
        file_path = Path(directory, "task_specifications.pkl")
        with open(file_path, "wb") as fd:
            pickle.dump(self, fd, protocol=4)

    # def to_primitives(self):
    #     d = deepcopy(self.__dict__)
    #     for i, band_info in enumerate(self.bands_info):
    #         d["bands_info"][i] = band_info.to_primitives()

    #     d["label_type"] = self.label_type.to_primitives()
    #     return d

    # def from_dict(self, object_dict):
    #     self.__dict__ = object_dict
    #     for i, band_info in enumerate(self.bands_info):
    #         self.bands_info[i] = io.from_primitives(*band_info)
    #     self.label_type = io.from_primitives(*self.label_type)

