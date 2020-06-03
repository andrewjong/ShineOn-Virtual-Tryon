# coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json

from datasets.cp_dataset import CPDataset
from datasets.mpv_dataset import MPVDataset
from datasets.vvt_dataset import VVTDataset


class CpVvtMpvDataset(data.Dataset):
    """Combines datasets
    """

    def name(self):
        return "CpVvtMpvDataset"

    def __init__(self, opt):
        super(CpVvtMpvDataset, self).__init__()
        # base setting
        self.opt = opt

        self.cp_dataset = CPDataset(opt)
        self.vvt_dataset = VVTDataset(opt)
        self.mpv_dataset = MPVDataset(opt)

    def __getitem__(self, index):
        if index < len(self.cp_dataset):
            return self.cp_dataset[index]

        index -= len(self.cp_dataset)
        if index < len(self.vvt_dataset):
            return self.vvt_dataset[index]

        index -= len(self.vvt_dataset)
        return self.mpv_dataset[index]

    def __len__(self):
        return len(self.cp_dataset) + len(self.vvt_dataset) + len(self.mpv_dataset)
