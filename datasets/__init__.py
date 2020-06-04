from .cp_dataset import CPDataset, CPDataLoader
from .cp_vvt_mpv_dataset import CpVvtMpvDataset
from .vvt_dataset import VVTDataset
from .mpv_dataset import MPVDataset

DATASETS = {
    "cp": CPDataset,
    "cp_vvt_mpv": CpVvtMpvDataset,
    "vvt": VVTDataset,
    "mpv": MPVDataset
}

def get_dataset_class(name):
    return DATASETS[name]
