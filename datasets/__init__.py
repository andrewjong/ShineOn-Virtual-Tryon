from .cp_dataset import CPDataset, CPDataLoader
from .cp_vvt_mpv_dataset import CpVvtMpvDataset
from .vvt_dataset import VVTDataset
from .mpv_dataset import MPVDataset
from .vvt_list_dataset import VVTListDataset

DATASETS = {
    "cp": CPDataset,
    "cp_vvt_mpv": CpVvtMpvDataset,
    "vvt": VVTDataset,
    "mpv": MPVDataset,
    "vvt_list": VVTListDataset
}

def get_dataset_class(name):
    return DATASETS[name]
