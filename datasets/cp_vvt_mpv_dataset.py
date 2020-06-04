# coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json

from datasets.cp_dataset import CPDataset, CPDataLoader
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
            item = self.cp_dataset[index]
            return item

        index -= len(self.cp_dataset)
        if index < len(self.vvt_dataset):
            item = self.vvt_dataset[index]
            return item

        index -= len(self.vvt_dataset)
        item = self.mpv_dataset[index]
        return item

    def __len__(self):
        return len(self.cp_dataset) + len(self.vvt_dataset) + len(self.mpv_dataset)


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--shuffle", action="store_true", help="shuffle input data")
    parser.add_argument("-b", "--batch-size", type=int, default=4)
    parser.add_argument("-j", "--workers", type=int, default=1)

    opt = parser.parse_args()
    dataset = CpVvtMpvDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print(
        "Size of the dataset: %05d, dataloader: %04d"
        % (len(dataset), len(data_loader.data_loader))
    )
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed

    embed()
