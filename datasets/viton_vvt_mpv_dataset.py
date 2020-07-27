# coding=utf-8
import torch.utils.data as data
import torchvision.transforms as transforms

from datasets.viton_dataset import VitonDataset
from datasets.cpvton_dataset import CPDataLoader
from datasets.mpv_dataset import MPVDataset
from datasets.vvt_dataset import VVTDataset


class VitonVvtMpvDataset(data.Dataset):
    """Combines datasets
    """

    def name(self):
        return "VitonVvtMpvDataset"

    def __init__(self, opt):
        super(VitonVvtMpvDataset, self).__init__()
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
