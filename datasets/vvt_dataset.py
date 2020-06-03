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
        self.root = opt.vvt_dataroot  # override this
        del self.data_list  # not using this
        del self.data_path
        self.CLOTH_THRESH = 240

    def _load_file_paths(self):
        """ Reads the datalist txt file for CP-VTON"""
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

    def _get_input_cloth_path(self, index):
        image_path = self.image_names[index]
        folder_id = VVTDataset.extract_folder_id(image_path)
        cloth_folder = osp.join(self.root, "lip_clothes_person", folder_id)
        cloth_path = [f for f in os.listdir(cloth_folder) if "cloth" in f][0]
        return cloth_path

    def get_input_cloth_mask(self, input_cloth: torch.Tensor):
        """ Creates a mask directly from the input_cloth """
        cloth_mask = input_cloth.clone().detach()
        t = self.CLOTH_THRESH
        # make the mask
        cloth_mask[
            input_cloth[0] >= t and input_cloth[1] >= t and input_cloth[2] >= t
        ] = 0
        cloth_mask[
            input_cloth[0] >= t and input_cloth[1] >= t and input_cloth[2] >= t
        ] = 1
        return cloth_mask

    ########################
    # PERSON REPRESENTATION
    ########################
    def _get_person_image_name(self, index):
        image_path = self._get_person_image_path(index)
        id = VVTDataset.extract_folder_id(image_path)
        name = osp.join(id, osp.basename(image_path))
        return name

    def _get_person_image_path(self, index):
        # because we globbed, the path is the list
        return self.image_names[index]

    def _get_person_parsed_path(self, index):
        image_path = self._get_person_image_path(index)
        folder = f"lip_{self.opt.datamode}_frames_parsing"
        id = VVTDataset.extract_folder_id(image_path)
        parsed_fname = os.path.split(image_path)[-1].replace(".png", "_label.png")
        parsed_path = osp.join(self.root, folder, id, parsed_fname)
        return parsed_path

    def get_input_person_pose_path(self, index):
        image_path = self.image_names[index]
        folder = f"lip_{self.opt.datamode}_frames"
        id = VVTDataset.extract_folder_id(image_path)

        keypoint_fname = os.path.split(image_path)[-1].replace(
            ".png", "_keypoints.json"
        )

        pose_path = osp.join(self.root, folder, id, keypoint_fname)
        return pose_path

