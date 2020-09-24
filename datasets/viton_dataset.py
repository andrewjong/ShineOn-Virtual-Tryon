import argparse
import os.path as osp

from datasets.tryon_dataset import TryonDataset


class VitonDataset(TryonDataset):
    """ CP-VTON dataset with the original Viton folder structure """

    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser, is_train, shared=False):
        if not shared:
            parser = super(VitonDataset, VitonDataset).modify_commandline_options(
                parser, is_train
            )
        parser.add_argument("--viton_dataroot", default="data")
        parser.add_argument("--data_list", default="train_pairs.txt")
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.data_list = opt.data_list
        self.data_path = osp.join(opt.viton_dataroot, opt.datamode)

    # @overrides
    def load_file_paths(self, i_am_validation=False):
        """
        Reads the datalist txt file for CP-VTON
        sets self.image_names and self.cloth_names. they should correspond 1-to-1
        """
        self.root = self.opt.viton_dataroot
        im_names = []
        c_names = []
        with open(osp.join(self.root, self.opt.data_list), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.image_names = im_names
        self.cloth_names = c_names

    ########################
    # CLOTH REPRESENTATION
    ########################
    # @overrides
    def get_input_cloth_path(self, index):
        """
        Get the file path for the product image input.
        Called by get_input_cloth()
        """
        c_name = self.get_input_cloth_name(index)
        folder = "cloth" if self.opt.model == "warp" else "warp-cloth"
        cloth_path = osp.join(self.data_path, folder, c_name)
        return cloth_path

    # @overrides
    def get_input_cloth_name(self, index):
        # determines the written thing
        return self.cloth_names[index]

    ########################
    # PERSON REPRESENTATION
    ########################

    # @overrides
    def get_person_image_name(self, index):
        """ basename of the image file """
        return self.image_names[index]

    # @overrides
    def get_person_image_path(self, index):
        im_name = self.get_person_image_name(index)
        image_path = osp.join(self.data_path, "image", im_name)
        return image_path

    # @overrides
    def get_person_parsed_path(self, index):
        """ path of the clothing seguemtnation """
        im_name = self.get_person_image_name(index)
        parse_name = im_name.replace(".jpg", ".png")
        parsed_path = osp.join(self.data_path, "image-parse", parse_name)
        return parsed_path

    # @overrides
    def get_person_cocopose_path(self, index):
        """ path to pose keypoints """
        im_name = self.get_person_image_name(index)
        _pose_name = im_name.replace(".jpg", "_keypoints.json")
        pose_path = osp.join(self.data_path, "pose", _pose_name)
        return pose_path

    def get_person_flow_path(self, index):
        return NotImplementedError("THIS IS TODO. Image datasets don't have flow")

    def get_person_densepose_path(self, index):
        return NotImplementedError("THIS IS TODO. For now use cocopose on VITON")