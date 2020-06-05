# coding=utf-8
import os
import os.path as osp

import torch

from datasets.cp_dataset import CPDataLoader, CPDataset
from datasets.vvt_dataset import VVTDataset


class MPVDataset(VVTDataset):
    """Dataset for CP-VTON. """

    def __init__(self, opt):
        super(MPVDataset, self).__init__(opt)

    def load_file_paths(self):
        """ Reads the datalist txt file for CP-VTON"""
        self.root = self.opt.mpv_dataroot
        self.image_names = []
        self.cloth_names = []

        datalist = osp.join(self.root, "all_poseA_poseB_clothes_0607.txt")
        with open(datalist, "r") as f:
            for line in f.readlines():
                person_name_1, person_name_2, cloth_name, _ = line.strip().split()
                self.image_names.extend([person_name_1, person_name_2])
                # both poses correspond to the same cloth
                self.cloth_names.extend([cloth_name, cloth_name])

        assert len(self.image_names) == len(
            self.cloth_names
        ), f"len mismatch: {len(self.image_names)} != {len(self.cloth_names)}"

    ########################
    # CLOTH REPRESENTATION
    ########################
    def get_input_cloth_name(self, index):
        return self.cloth_names[index]

    def get_input_cloth_path(self, index):
        cloth_name = self.get_input_cloth_name(index)
        subdir = "all" if self.stage == "GMM" else "warp-cloth"
        cloth_path = osp.join(self.root, subdir, cloth_name)
        return cloth_path

    ########################
    # PERSON REPRESENTATION
    ########################
    def get_person_image_name(self, index):
        return self.image_names[index]

    def get_person_image_path(self, index):
        image_name = self.get_person_image_name(index)
        image_path = osp.join(self.root, "all", image_name)
        return image_path

    def get_person_parsed_path(self, index):
        image_name = self.get_person_image_name(index).replace(".jpg", ".png")
        parsed_path = osp.join(self.root, "all_parsing", image_name)
        return parsed_path

    def get_input_person_pose_path(self, index):
        image_name = self.get_person_image_name(index)
        pose_path = osp.join(self.root, "all_person_clothes_keypoints", image_name)
        pose_path = pose_path.replace(".jpg", "_keypoints.json")
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
    dataset = MPVDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print(
        "Size of the dataset: %05d, dataloader: %04d"
        % (len(dataset), len(data_loader.data_loader))
    )
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed

    embed()
