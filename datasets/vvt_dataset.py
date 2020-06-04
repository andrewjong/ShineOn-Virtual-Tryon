# coding=utf-8
import torch
from glob import glob
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

from datasets.cp_dataset import CPDataset, CPDataLoader

import os
import os.path as osp
import numpy as np
import json


class VVTDataset(CPDataset):
    """Dataset for CP-VTON. """

    def name(self):
        return "CPDataset"

    def __init__(self, opt):
        super(VVTDataset, self).__init__(opt)
        del self.data_list  # not using this
        del self.data_path
        self.CLOTH_THRESH = 240

    def load_file_paths(self):
        """ Reads the datalist txt file for CP-VTON"""
        self.root = self.opt.vvt_dataroot  # override this
        folder = f"lip_{self.opt.datamode}_frames"
        self.image_names = glob(f"{self.root}/{folder}/**/*.png")

    @staticmethod
    def extract_folder_id(image_path):
        return image_path.split(os.sep)[-2]

    ########################
    # CLOTH REPRESENTATION
    ########################
    def get_cloth_representation(self, index):
        cloth = self.get_input_cloth(index)
        cloth_mask = self.get_input_cloth_mask(cloth)
        return cloth, cloth_mask

    def get_input_cloth_path(self, index):
        image_path = self.image_names[index]
        folder_id = VVTDataset.extract_folder_id(image_path)
        cloth_folder = osp.join(self.root, "lip_clothes_person", folder_id)
        cloth_path = glob(f"{cloth_folder}/*cloth*")[0]
        return cloth_path

    def get_input_cloth_mask(self, input_cloth: torch.Tensor):
        """ Creates a mask directly from the input_cloth """
        # make the mask
        low = torch.zeros_like(input_cloth)
        high = torch.ones_like(input_cloth)
        cloth_mask = torch.where(input_cloth >= self.CLOTH_THRESH, low, high)
        cloth_mask = cloth_mask[0].unsqueeze(0)  # the mask should be a single channel
        return cloth_mask

    def get_input_cloth_name(self, index):
        return ""

    ########################
    # PERSON REPRESENTATION
    ########################
    def get_person_image_name(self, index):
        image_path = self.get_person_image_path(index)
        id = VVTDataset.extract_folder_id(image_path)
        name = osp.join(id, osp.basename(image_path))
        return name

    def get_person_image_path(self, index):
        # because we globbed, the path is the list
        return self.image_names[index]

    def get_person_parsed_path(self, index):
        image_path = self.get_person_image_path(index)
        folder = f"lip_{self.opt.datamode}_frames_parsing"
        id = VVTDataset.extract_folder_id(image_path)
        parsed_fname = os.path.split(image_path)[-1].replace(".png", "_label.png")
        parsed_path = osp.join(self.root, folder, id, parsed_fname)
        return parsed_path

    def get_input_person_pose_path(self, index):
        image_path = self.image_names[index]
        folder = f"lip_{self.opt.datamode}_frames_keypoint"
        id = VVTDataset.extract_folder_id(image_path)

        keypoint_fname = os.path.split(image_path)[-1].replace(
            ".png", "_keypoints.json"
        )

        pose_path = osp.join(self.root, folder, id, keypoint_fname)
        return pose_path


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
    dataset = VVTDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print(
        "Size of the dataset: %05d, dataloader: %04d"
        % (len(dataset), len(data_loader.data_loader))
    )
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed

    embed()
