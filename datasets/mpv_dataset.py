# coding=utf-8
import argparse
import os.path as osp

from datasets.tryon_dataset import TryonDataset


class MPVDataset(TryonDataset):
    """ CP-VTON dataset with the MPV folder structure. """

    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser, is_train, shared=False):
        if not shared:
            parser = super(MPVDataset, MPVDataset).modify_commandline_options(
                parser, is_train
            )
        parser.add_argument("--mpv_dataroot", default="/data_hdd/mpv_competition")
        return parser

    def __init__(self, opt):
        super(MPVDataset, self).__init__(opt)

    # @overrides(TryonDataset)
    def load_file_paths(self, i_am_validation=False):
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
    # @overrides(TryonDataset)
    def get_input_cloth_path(self, index):
        cloth_name = self.get_input_cloth_name(index)
        subdir = "all" if self.opt.model == "warp" else "warp-cloth"
        cloth_path = osp.join(self.root, subdir, cloth_name)
        return cloth_path

    # @overrides(TryonDataset)
    def get_input_cloth_name(self, index):
        return self.cloth_names[index]

    ########################
    # PERSON REPRESENTATION
    ########################
    # @overrides(TryonDataset)
    def get_person_image_path(self, index):
        image_name = self.get_person_image_name(index)
        image_path = osp.join(self.root, "all", image_name)
        return image_path

    # @overrides(TryonDataset)
    def get_person_image_name(self, index):
        return self.image_names[index]

    # @overrides(TryonDataset)
    def get_person_parsed_path(self, index):
        image_name = self.get_person_image_name(index).replace(".jpg", ".png")
        parsed_path = osp.join(self.root, "all_parsing", image_name)
        return parsed_path

    # @overrides(TryonDataset)
    def get_person_cocopose_path(self, index):
        image_name = self.get_person_image_name(index)
        pose_path = osp.join(self.root, "all_person_clothes_keypoints", image_name)
        pose_path = pose_path.replace(".jpg", "_keypoints.json")
        return pose_path

    def get_person_densepose_path(self, index):
        return NotImplementedError("THIS IS TODO. For now use cocopose on MPV")

    def get_person_flow_path(self, index):
        return NotImplementedError("THIS IS TODO. Image datasets don't have flow")
