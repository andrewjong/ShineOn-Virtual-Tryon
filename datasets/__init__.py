import importlib

from .base_dataset import BaseDataset
from .cpvton_dataset import CpVtonDataset, CPDataLoader
from .mpv_dataset import MPVDataset
from .viton_dataset import VitonDataset
from .viton_vvt_mpv_dataset import VitonVvtMpvDataset
from .vvt_dataset import VVTDataset
from .vvt_list_dataset import VVTListDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (dataset_filename, target_dataset_name)
        )

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options
