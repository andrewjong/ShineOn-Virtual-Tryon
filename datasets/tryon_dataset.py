# coding=utf-8
import json
from abc import abstractmethod, ABC
from argparse import ArgumentParser
import os


import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw

from datasets import BaseDataset
from datasets.util import segment_cloths_from_image

from models.flownet2_pytorch.utils.flow_utils import flow2img, readFlow



class TryonDataset(BaseDataset, ABC):
    """ Loads all the necessary items for CP-Vton """

    RGB_CHANNELS = 3

    COCOPOSE_CHANNELS = 18
    IM_HEAD_CHANNELS = RGB_CHANNELS
    SILHOUETTE_CHANNELS = 1

    AGNOSTIC_CHANNELS = COCOPOSE_CHANNELS + IM_HEAD_CHANNELS + SILHOUETTE_CHANNELS

    CLOTH_CHANNELS = RGB_CHANNELS
    CLOTH_MASK_CHANNELS = 1

    DENSEPOSE_CHANNELS = 3

    #FLOW_CHANNELS = 2


    @staticmethod
    def modify_commandline_options(parser: ArgumentParser, is_train):
        parser.add_argument(
            "--cloth_mask_threshold",
            type=int,
            default=240,
            help="threshold to remove white background for the cloth mask; "
            "everything above this value is removed [0-255].",
        )
        parser.add_argument(
            "--image_scale", type=float, default=1, help="first scale to this"
        )
        parser.add_argument(
            "--fine_width", type=int, default=192, help="then crop to this"
        )
        parser.add_argument(
            "--fine_height", type=int, default=256, help="then crop to this"
        )
        parser.add_argument("--radius", type=int, default=5)
        return parser

    def __init__(self, opt):
        super(TryonDataset, self).__init__(opt)
        self.cloth_mask_threshold = opt.cloth_mask_threshold
        # base setting
        self.opt = opt
        self.datamode = opt.datamode  # train or test or self-defined
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.center_crop = transforms.CenterCrop((self.fine_height, self.fine_width))
        self.to_tensor_and_norm_rgb = transforms.Compose(
            [
                self.center_crop,
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.to_tensor_and_norm_gray = transforms.Compose(
            [
                self.center_crop,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.flow_norm = transforms.Normalize((0.5, 0.5), (0.5, 0.5))

        self.image_names = []
        # load data list
        self.load_file_paths()

    @abstractmethod
    def load_file_paths(self):
        """
        Reads the datalist txt file for CP-VTON
        sets self.image_names and self.cloth_names. they should correspond 1-to-1
        """
        pass

    def __len__(self):
        return len(self.image_names)

    def open_image_as_normed_tensor(self, path):
        img = Image.open(path)
        tensor = self.to_tensor_and_norm_rgb(img)
        return tensor

    ########################
    # CLOTH REPRESENTATION
    ########################

    def get_cloth_representation(self, index):
        """
        call all cloth loaders
        :param index:
        :return: cloth, cloth_mask
        """
        cloth = self.get_input_cloth(index)
        cloth_mask = self.get_input_cloth_mask(cloth)
        return {"cloth": cloth, "cloth_mask": cloth_mask}

    def get_input_cloth_mask(self, input_cloth: torch.Tensor):
        """ Creates a mask directly from the input_cloth """
        # make the mask
        low = torch.zeros_like(input_cloth)
        high = torch.ones_like(input_cloth)
        cloth_mask = torch.where(input_cloth >= self.cloth_mask_threshold, low, high)
        cloth_mask = cloth_mask[0].unsqueeze(0)  # the mask should be a single channel
        return cloth_mask

    def get_input_cloth(self, index):
        """
        Calls _get_input_cloth_path() for a file path, then opens that path as an image
        tensor
        """
        cloth_path = self.get_input_cloth_path(index)
        c = self.open_image_as_normed_tensor(cloth_path)
        return c

    @abstractmethod
    def get_input_cloth_path(self, index):
        """
        Get the file path for the product image input.
        Called by get_input_cloth()
        """
        pass

    @abstractmethod
    def get_input_cloth_name(self, index):
        # determines the written thing
        pass

    ########################
    # PERSON REPRESENTATION
    ########################

    def get_person_representation(self, index):
        """
        get all person represetations
        :param index:
        :return:
        """
        ret = {}
        # person image
        image = self.get_person_image(index)
        # load parsing image
        _parse_array = self.get_person_parsed(index)
        # body silhouette
        silhouette = self.get_input_person_body_silhouette(_parse_array)
        # isolated head
        im_head = self.get_input_person_head(image, _parse_array)
        # isolated cloth
        im_cloth = segment_cloths_from_image(image, _parse_array)

        # load pose points
        _pose_map, im_cocopose = self.get_input_person_pose(index)

        if "agnostic" in self.opt.person_inputs:
            _agnostic_items = [silhouette, im_head, _pose_map]
            agnostic = torch.cat(_agnostic_items, 0)
            ret["agnostic"] = agnostic

        if "densepose" in self.opt.person_inputs:
            densepose = self.get_person_densepose(index)
            ret["densepose"] = densepose


        ret.update(
            {
                "silhouette": silhouette,
                "image": image,
                "im_head": im_head,
                "im_cloth": im_cloth,
                "im_cocopose": im_cocopose,
            }
        )
        return ret

    def get_person_image(self, index):
        """
        helper function to get the person image; not used as input to the network. used
        instead to form the other input
        :param index:
        :return:
        """
        # person image
        image_path = self.get_person_image_path(index)
        im = self.open_image_as_normed_tensor(image_path)
        return im


    def get_person_flow(self, index):
        """
        helper function to get the person image; not used as input to the network. used
        instead to form the other input
        :param index:
        :return:
        """
        # person image
        image_path = self.get_person_image_path(index)
        image_path = image_path.replace(".png", ".flo")
        image_path = image_path.replace(f"{self.opt.datamode}_frames", "optical_flow")
        image = readFlow(image_path)
        #image = flow2img(image)
        image = torch.from_numpy(image).permute(2, 0, 1)
        #image = self.center_crop(image)
        #print("flow", image.shape)
        image = self.flow_norm(image)
        #image = Image.open(image_path)
        #im = self.to_tensor_and_norm_gray(image)
        return image


    def get_person_densepose(self, index):
        """
        helper function to get the person image; not used as input to the network. used
        instead to form the other input
        :param index:
        :return:
        """
        # person image
        image_path = self.get_person_densepose_path(index)
        try:
            iuv = self.open_image_as_normed_tensor(image_path)
        except FileNotFoundError:
            iuv = torch.zeros(3, self.fine_height, self.fine_width)
        return iuv

    def get_person_parsed(self, index):
        """ loads parsing image """
        parsed_path = self.get_person_parsed_path(index)
        im_parse = Image.open(parsed_path)
        im_parse = self.center_crop(im_parse)
        parse_array = np.array(im_parse)
        return parse_array

    def get_input_person_head(self, im, _parse_array):
        """ from cp-vton, get the floating head alone"""
        # ISOLATE HEAD. head parts, probably face, hair, sunglasses. combines into a 1d binary mask
        _parse_head = (
            (_parse_array == 1).astype(np.float32)
            + (_parse_array == 2).astype(np.float32)
            + (_parse_array == 4).astype(np.float32)
            + (_parse_array == 13).astype(np.float32)
        )
        _phead = torch.from_numpy(_parse_head)  # [0,1]
        im_h = im * _phead - (1 - _phead)  # [-1,1], fill 0 for other parts
        return im_h

    def get_input_person_body_silhouette(self, _parse_array):
        """ from cp-vton, the body silhouette """
        # ISOLATE BODY SHAPE
        # removes the background
        _parse_shape = (_parse_array > 0).astype(np.float32)
        # silhouette downsample, reduces resolution, makes the silhouette "blurry"
        _parse_shape = Image.fromarray((_parse_shape * 255).astype(np.uint8))
        _parse_shape = _parse_shape.resize(
            (self.fine_width // 16, self.fine_height // 16), Image.BILINEAR
        )
        _parse_shape = _parse_shape.resize(
            (self.fine_width, self.fine_height), Image.BILINEAR
        )
        try:
            silhouette = self.to_tensor_and_norm_rgb(_parse_shape)  # [-1,1]
        except Exception as e1:
            # print("ERROR:", e1)
            silhouette = self.to_tensor_and_norm_gray(_parse_shape)
        except Exception as e2:
            raise e2

        return silhouette

    def get_input_person_pose(self, index):
        """from cp-vton, loads the pose as white squares
        returns pose map, image of pose map
        """
        pose_path = self.get_person_cocopose_path(index)
        with open(pose_path, "r") as f:
            pose_label = json.load(f)
            try:
                pose_data = pose_label["people"][0]["pose_keypoints"]
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))
            except IndexError:
                # print("Warning: No pose data found for", pose_path)
                pose_data = None

        pose_map, im_cocopose = self.convert_pose_data_to_pose_map_and_vis(pose_data)

        return pose_map, im_cocopose

    def convert_pose_data_to_pose_map_and_vis(self, pose_data):
        """
        Reads a pose data array and makes a 1-hot tensor and visualization

        Args:
            pose_data: an NUM_KEYPOINTS x 3 array of pose points (first 2 are x, y; last is confidence)

        Returns: 1 hot tensor pose map; 1 channel visualization

        """
        point_num = (
            pose_data.shape[0] if pose_data is not None else 18
        )  # how many pose joints
        pose_map = torch.zeros(
            point_num, self.fine_height, self.fine_width
        )  # constructs an N-channel map

        im_cocopose = Image.new("L", (self.fine_width, self.fine_height))

        if pose_data is not None:
            pose_draw = ImageDraw.Draw(im_cocopose)
            # draws a big white square around the joint on the appropriate channel. I guess this emphasizes it
            r = self.radius
            for i in range(point_num):
                one_map = Image.new("L", (self.fine_width, self.fine_height))

                try:
                    one_map_tensor = self.to_tensor_and_norm_rgb(one_map)  # [-1,1]
                except Exception as e1:
                    # print("ERROR:", e1)
                    one_map_tensor = self.to_tensor_and_norm_gray(one_map)
                except Exception as e2:
                    raise e2

                pose_map[i] = one_map_tensor[0]

                draw = ImageDraw.Draw(one_map)
                pointx = pose_data[i, 0]
                pointy = pose_data[i, 1]
                if pointx > 1 and pointy > 1:
                    draw.rectangle(
                        (pointx - r, pointy - r, pointx + r, pointy + r),
                        "white",
                        "white",
                    )
                    pose_draw.rectangle(
                        (pointx - r, pointy - r, pointx + r, pointy + r),
                        "white",
                        "white",
                    )
        # just for visualization
        try:
            im_cocopose = self.to_tensor_and_norm_rgb(im_cocopose)  # [-1,1]
        except Exception as e1:
            # print("ERROR:", e1)
            im_cocopose = self.to_tensor_and_norm_gray(im_cocopose)
        except Exception as e2:
            raise e2

        return pose_map, im_cocopose

    @abstractmethod
    def get_person_image_path(self, index):
        pass

    @abstractmethod
    def get_person_image_name(self, index):
        """ basename of the image file """
        pass

    @abstractmethod
    def get_person_cocopose_path(self, index):
        """ path to pose keypoints """
        pass

    @abstractmethod
    def get_person_parsed_path(self, index):
        """ path of the clothing seguemtnation """
        pass

    @abstractmethod
    def get_person_densepose_path(self, index):
        pass

    ########################
    # getitem
    ########################

    def __getitem__(self, index):
        # grid visualization for warping
        grid_vis = (
            self.open_image_as_normed_tensor("grid.png")
            if self.opt.model == "warp"
            else ""
        )
        result = {
            "dataset_name": self.__class__.__name__,
            "cloth_name": self.get_input_cloth_name(index),  # for visualization
            "cloth_path": self.get_input_cloth_path(index),
            "image_name": self.get_person_image_name(index),
            "image_path": self.get_person_image_path(index),
            "grid_vis": grid_vis,
        }

        if self.opt.flow and self.opt.model == "unet_mask":
            #print("flow")
            flow = self.get_person_flow(index) #torch.zeros(2, self.opt.fine_height, self.opt.fine_width)
            result["flow"] = flow

        # cloth representation
        result.update(self.get_cloth_representation(index))
        # person representation
        result.update(self.get_person_representation(index))

        # def run_assertions():
        #     assert cloth.shape == torch.Size(
        #         [3, 256, 192]
        #     ), f"cloth.shape = {cloth.shape} on {im_path} {cloth_path}"
        #     assert cloth_mask.shape == torch.Size(
        #         [1, 256, 192]
        #     ), f"cloth_mask.shape = {cloth_mask.shape} on {im_path} {cloth_path}"
        #     assert im.shape == torch.Size([3, 256, 192]), f"im = {im.shape} on {im_path}"
        #     assert im_cloth.shape == torch.Size(
        #         [3, 256, 192]
        #     ), f"im_cloth = {im_cloth} on {im_name}"
        #     assert im_cocopose.shape == torch.Size(
        #         [1, 256, 192]
        #     ), f"im_cocopose.shape = {im_cocopose.shape} on {im_path}"
        #     assert silhouette.shape == torch.Size(
        #         [1, 256, 192]
        #     ), f"silhouette.shape = {silhouette.shape} on {im_path}"
        #     assert im_head.shape == torch.Size(
        #         [3, 256, 192]
        #     ), f"im_head = {im_head.shape} on {im_path}"
        #     assert agnostic.shape == torch.Size(
        #         # silhouette, im_head, cocopose, densepose
        #         [1 + 3 + 18 + 3, 256, 192]
        #     ), f"agnostic = {agnostic.shape} on {im_path}"
        #     if im_grid is not "":
        #         assert im_grid.shape == torch.Size(
        #             [3, 256, 192]
        #         ), f"im_grid = {im_grid.shape} on {im_path}"
        # run_assertions()

        return result
