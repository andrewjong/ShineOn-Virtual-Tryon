from .cpvton_dataset import CpVtonDataset, CPDataLoader
from .viton_vvt_mpv_dataset import VitonVvtMpvDataset
from .viton_dataset import VitonDataset
from .vvt_dataset import VVTDataset
from .mpv_dataset import MPVDataset
from .vvt_list_dataset import VVTListDataset

DATASETS = {
    "viton_vvt_mpv": VitonVvtMpvDataset,
    "viton": VitonDataset,
    "vvt": VVTDataset,
    "mpv": MPVDataset,
    "vvt_list": VVTListDataset
}

def get_dataset_class(name):
    return DATASETS[name]
