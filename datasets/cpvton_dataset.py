# coding=utf-8
from abc import ABC, abstractmethod
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json

from datasets.util import segment_cloths_from_image


class CpVtonDataset(ABC, data.Dataset):
    """ Loads all the necessary items for CP-Vton """

    def __init__(self, opt):
        super(CpVtonDataset, self).__init__()
        self.CLOTH_THRESH = 240
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode  # train or test or self-defined
        self.stage = opt.stage  # GMM or TOM
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.center_crop = transforms.CenterCrop((self.fine_height, self.fine_width))
        self.to_tensor_and_norm = transforms.Compose(
            [
                self.center_crop,
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
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
        return cloth, cloth_mask

    def get_input_cloth_mask(self, input_cloth: torch.Tensor):
        """ Creates a mask directly from the input_cloth """
        # make the mask
        low = torch.zeros_like(input_cloth)
        high = torch.ones_like(input_cloth)
        cloth_mask = torch.where(input_cloth >= self.CLOTH_THRESH, low, high)
        cloth_mask = cloth_mask[0].unsqueeze(0)  # the mask should be a single channel
        return cloth_mask

    def get_input_cloth(self, index):
        """
        Calls _get_input_cloth_path() for a file path, then opens that path as an image
        tensor
        """
        cloth_path = self.get_input_cloth_path(index)
        c = Image.open(cloth_path)
        c = self.to_tensor_and_norm(c)  # [-1,1]
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
        # person image
        im = self.get_person_image(index)
        # load parsing image
        _parse_array = self.get_person_parsed(index)
        # body silhouette
        silhouette = self.get_input_person_body_silhouette(_parse_array)
        # isolated head
        im_head = self.get_input_person_head(im, _parse_array)
        # isolated cloth
        im_cloth = segment_cloths_from_image(im, _parse_array)

        # load pose points
        _pose_map, im_pose = self.get_input_person_pose(index)

        # person-agnostic representation
        agnostic = torch.cat([silhouette, im_head, _pose_map], 0)

        return silhouette, im, im_head, im_cloth, im_pose, agnostic

    def get_person_image(self, index):
        """
        helper function to get the person image; not used as input to the network. used
        instead to form the other input
        :param index:
        :return:
        """
        # person image
        image_path = self.get_person_image_path(index)
        im = Image.open(image_path)
        im = self.to_tensor_and_norm(im)  # [-1,1]
        return im

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
        silhouette = self.to_tensor_and_norm(_parse_shape)  # [-1,1]
        return silhouette

    def get_input_person_pose(self, index):
        """from cp-vton, loads the pose as white squares
        returns pose map, image of pose map
        """
        pose_path = self.get_input_person_pose_path(index)
        with open(pose_path, "r") as f:
            pose_label = json.load(f)
            try:
                pose_data = pose_label["people"][0]["pose_keypoints"]
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))
            except IndexError:
                # print("Warning: No pose data found for", pose_path)
                pose_data = None

        pose_map, im_pose = self.convert_pose_data_to_pose_map_and_vis(pose_data)

        return pose_map, im_pose

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

        im_pose = Image.new("L", (self.fine_width, self.fine_height))

        if pose_data is not None:
            pose_draw = ImageDraw.Draw(im_pose)
            # draws a big white square around the joint on the appropriate channel. I guess this emphasizes it
            r = self.radius
            for i in range(point_num):
                one_map = Image.new("L", (self.fine_width, self.fine_height))
                one_map_tensor = self.to_tensor_and_norm(one_map)
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
        im_pose = self.to_tensor_and_norm(im_pose)
        return pose_map, im_pose

    @abstractmethod
    def get_person_image_path(self, index):
        pass

    @abstractmethod
    def get_person_image_name(self, index):
        """ basename of the image file """
        pass

    @abstractmethod
    def get_person_parsed_path(self, index):
        """ path of the clothing seguemtnation """
        pass

    @abstractmethod
    def get_input_person_pose_path(self, index):
        """ path to pose keypoints """
        pass

    ########################
    # getitem
    ########################

    def __getitem__(self, index):
        im_name = self.get_person_image_name(index)
        im_path = self.get_person_image_path(index)
        cloth_path = self.get_input_cloth_path(index)
        # cloth representation
        cloth, cloth_mask = self.get_cloth_representation(index)

        # person representation
        (
            silhouette,
            im,
            im_head,
            im_cloth,
            im_pose,
            agnostic,
        ) = self.get_person_representation(index)

        # grid visualization for warping
        if self.stage == "GMM":
            im_grid = Image.open("grid.png")
            im_grid = self.to_tensor_and_norm(im_grid)
        else:
            im_grid = ""

        def run_assertions():
            assert cloth.shape == torch.Size(
                [3, 256, 192]
            ), f"cloth.shape = {cloth.shape} on {im_path} {cloth_path}"
            assert cloth_mask.shape == torch.Size(
                [1, 256, 192]
            ), f"cloth_mask.shape = {cloth_mask.shape} on {im_path} {cloth_path}"
            assert im.shape == torch.Size([3, 256, 192]), f"im = {im.shape} on {im_path}"
            assert im_cloth.shape == torch.Size(
                [3, 256, 192]
            ), f"im_cloth = {im_cloth} on {im_name}"
            assert im_pose.shape == torch.Size(
                [1, 256, 192]
            ), f"im_pose.shape = {im_pose.shape} on {im_path}"
            assert silhouette.shape == torch.Size(
                [1, 256, 192]
            ), f"silhouette.shape = {silhouette.shape} on {im_path}"
            assert im_head.shape == torch.Size(
                [3, 256, 192]
            ), f"im_head = {im_head} on {im_path}"
            assert agnostic.shape == torch.Size(
                [22, 256, 192]
            ), f"agnostic = {agnostic} on {im_path}"
            if im_grid is not "":
                assert im_grid.shape == torch.Size(
                    [3, 256, 192]
                ), f"im_grid = {im_grid} on {im_path}"
        run_assertions()

        result = {
            "dataset_name": self.__class__.__name__,
            "c_name": self.get_input_cloth_name(index),  # for visualization
            "c_path": cloth_path,
            "im_name": im_name,
            "cloth": cloth,  # for input
            "cloth_mask": cloth_mask,  # for input
            "image": im,  # for visualization
            "agnostic": agnostic,  # for input
            "parse_cloth": im_cloth,  # for ground truth
            "shape": silhouette,  # for visualization
            "head": im_head,  # for visualization
            "pose_image": im_pose,  # for visualization
            "grid_image": im_grid,  # for visualization
        }

        return result


class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=(train_sampler is None),
            num_workers=opt.workers,
            pin_memory=True,
            sampler=train_sampler,
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


