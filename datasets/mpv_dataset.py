# coding=utf-8
import os
import os.path as osp

import torch

from datasets.cp_dataset import CPDataLoader
from datasets.vvt_dataset import VVTDataset


class MPVDataset(VVTDataset):
    """Dataset for CP-VTON. """

    def name(self):
        return "MPVDataset"

    def __init__(self, opt):
        super(MPVDataset, self).__init__(opt)

    def _load_file_paths(self):
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
    def _get_input_cloth_name(self, index):
        return self.cloth_names[index]

    def _get_input_cloth_path(self, index):
        cloth_name = self._get_input_cloth_name(index)
        cloth_path = osp.join(self.root, "all", cloth_name)
        return cloth_path

    ########################
    # PERSON REPRESENTATION
    ########################
    def _get_person_image_name(self, index):
        return self.image_names[index]

    def _get_person_image_path(self, index):
        image_name = self._get_person_image_name(index)
        image_path = osp.join(self.root, "all", image_name)
        return image_path

    def _get_person_parsed_path(self, index):
        image_name = self._get_person_image_name(index)
        parsed_path = osp.join(self.root, "all_parsing", image_name)
        return parsed_path

    def get_input_person_pose_path(self, index):
        image_name = self._get_person_image_name(index)
        pose_path = osp.join(self.root, "all_person_clothes_keypoints", image_name)
        pose_path.replace(".png", "_keypoints.json")
        return pose_path


