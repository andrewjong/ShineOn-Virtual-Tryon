# coding=utf-8
from argparse import ArgumentParser

import torchvision.transforms as transforms

from datasets import BaseDataset
from datasets.mpv_dataset import MPVDataset
from datasets.viton_dataset import VitonDataset
from datasets.vvt_dataset import VVTDataset
from options.train_options import TrainOptions


class VitonVvtMpvDataset(BaseDataset):
    """Combines datasets
    """

    @staticmethod
    def modify_commandline_options(parser: ArgumentParser, is_train):
        parser = VitonDataset.modify_commandline_options(parser, is_train)
        parser = VVTDataset.modify_commandline_options(parser, is_train)
        parser = MPVDataset.modify_commandline_options(parser, is_train)
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

    def __getitem__(self, index):
        if index < len(self.viton_dataset):
            item = self.viton_dataset[index]
            return item

        index -= len(self.viton_dataset)
        if index < len(self.vvt_dataset):
            item = self.vvt_dataset[index]
            return item

        index -= len(self.vvt_dataset)
        item = self.mpv_dataset[index]
        return item

    def __len__(self):
        return len(self.viton_dataset) + len(self.vvt_dataset) + len(self.mpv_dataset)


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")

    opt = TrainOptions().parse()

    dataset = VitonVvtMpvDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print(
        "Size of the dataset: %05d, dataloader: %04d"
        % (len(dataset), len(data_loader.data_loader))
    )
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed

    embed()
