# coding=utf-8
from argparse import ArgumentParser

import torchvision.transforms as transforms

from datasets import BaseDataset
from datasets.mpv_dataset import MPVDataset
from datasets.n_frames_interface import maybe_combine_frames_and_channels
from datasets.tryon_dataset import TryonDatasetType
from datasets.viton_dataset import VitonDataset
from datasets.vvt_dataset import VVTDataset


class VitonVvtMpvDataset(BaseDataset):
    """Combines datasets
    """

    @staticmethod
    def modify_commandline_options(parser: ArgumentParser, is_train):
        parser = VVTDataset.modify_commandline_options(parser, is_train)
        parser = VitonDataset.modify_commandline_options(parser, is_train, shared=True)
        parser = MPVDataset.modify_commandline_options(parser, is_train, shared=True)
        return parser

    def name(self):
        return "VitonVvtMpvDataset"

    def __init__(self, opt):
        super(VitonVvtMpvDataset, self).__init__(opt)
        # base setting
        self.opt = opt

        self.viton_dataset = VitonDataset(opt)
        self.vvt_dataset = VVTDataset(opt)
        self.mpv_dataset = MPVDataset(opt)

        self.transforms = transforms.Compose([])

    @classmethod
    def make_validation_dataset(self, opt) -> TryonDatasetType:
        val = VVTDataset(opt, i_am_validation=True)
        return val

    def __getitem__(self, index):
        if index < len(self.viton_dataset):
            item = self.viton_dataset[index]
            return item
        index -= len(self.viton_dataset)

        if index < len(self.vvt_dataset):
            item = self.vvt_dataset[index]
            if self.opt.model == "warp":
                assert self.opt.n_frames_total == 1, (
                    f"{self.opt.n_frames_total=}; "
                    f"warp model shouldn't be using n_frames_total > 1"
                )
                item = maybe_combine_frames_and_channels(self.opt, item, has_batch_dim=False)
            return item
        index -= len(self.vvt_dataset)

        item = self.mpv_dataset[index]
        return item

    def __len__(self):
        return len(self.viton_dataset) + len(self.vvt_dataset) + len(self.mpv_dataset)
