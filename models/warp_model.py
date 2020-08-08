""" Also known as GMM """
import os.path as osp
from argparse import ArgumentParser

import torch.nn as nn
import torch.nn.functional as F

from datasets.n_frames_interface import maybe_combine_frames_and_channels
from datasets.tryon_dataset import TryonDataset
from models.base_model import BaseModel, parse_channels
from models.networks.cpvton.warp import (
    FeatureExtraction,
    FeatureL2Norm,
    FeatureCorrelation,
    FeatureRegression,
    TpsGridGen,
)
from visualization import tensor_list_for_board, get_save_paths, save_images


# coding=utf-8


class WarpModel(BaseModel):
    """ Geometric Matching Module """

    @classmethod
    def modify_commandline_options(cls, parser: ArgumentParser, is_train):
        parser = ArgumentParser(parents=[parser], add_help=False)
        parser = super(WarpModel, cls).modify_commandline_options(parser, is_train)
        parser.add_argument("--ngf", type=int, default=64)
        parser.add_argument("--grid_size", type=int, default=5)
        parser.set_defaults(inputs=("agnostic", "densepose"))
        parser.add_argument("--inputs_B", default="cloth")
        return parser

    def __init__(self, hparams):
        super(WarpModel, self).__init__(hparams)
        # n_frames = opt.n_frames if hasattr(opt, "n_frames") else 1
        self.extractionA = FeatureExtraction(
            self.in_channels, ngf=hparams.ngf, n_layers=3, norm_layer=nn.BatchNorm2d,
        )
        B_channels = parse_channels(hparams.inputs_B)
        self.extractionB = FeatureExtraction(
            B_channels, ngf=hparams.ngf, n_layers=3, norm_layer=nn.BatchNorm2d
        )
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(
            input_nc=192, output_dim=2 * hparams.grid_size ** 2
        )
        self.gridGen = TpsGridGen(
            hparams.fine_height, hparams.fine_width, grid_size=hparams.grid_size
        )

    def forward(self, inputA, inputB):
        featureA = self.extractionA(inputA)
        featureB = self.extractionB(inputB)
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
        correlation = self.correlation(featureA, featureB)

        theta = self.regression(correlation)
        grid = self.gridGen(theta)
        return grid, theta

    def training_step(self, batch, _):
        batch = maybe_combine_frames_and_channels(self.hparams, batch)
        # unpack
        agnostic = batch["agnostic"]
        c = batch["cloth"]
        im_c = batch["im_cloth"]
        im_g = batch["grid_vis"]

        # forward
        grid, theta = self.forward(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode="border")
        warped_grid = F.grid_sample(im_g, grid, padding_mode="zeros")
        # loss
        loss = F.l1_loss(warped_cloth, im_c)

        # Logging
        if self.global_step % self.hparams.display_count == 0:
            self.visualize(batch, warped_cloth, warped_grid)

        tensorboard_scalars = {"epoch": self.current_epoch, "loss": loss}

        return {"loss": loss, "log": tensorboard_scalars}

    def test_step(self, batch, batch_idx):
        dataset_names = batch["dataset_name"]
        # produce subfolders for each subdataset
        warp_cloth_dirs = [
            osp.join(self.test_results_dir, dname, "warp-cloth")
            for dname in dataset_names
        ]
        warp_mask_dirs = [
            osp.join(self.test_results_dir, dname, "warp-mask")
            for dname in dataset_names
        ]
        c_names = batch["cloth_name"]
        # if we already did a forward-pass on this batch, skip it
        save_paths = get_save_paths(c_names, warp_cloth_dirs)
        if all(osp.exists(s) for s in save_paths):
            progress_bar = {"file": f"Skipping {c_names[0]}"}
        else:
            progress_bar = {"file": c_names[0]}
            # unpack the the data
            agnostic = batch["agnostic"]
            c = batch["cloth"]
            cm = batch["cloth_mask"]
            im_g = batch["grid_vis"]

            # forward pass
            grid, theta = self.forward(agnostic, c)
            warped_cloth = F.grid_sample(c, grid, padding_mode="border")
            warped_mask = F.grid_sample(cm, grid, padding_mode="zeros")
            warped_grid = F.grid_sample(im_g, grid, padding_mode="zeros")

            # save images
            save_images(warped_cloth, c_names, warp_cloth_dirs)
            save_images(warped_mask * 2 - 1, c_names, warp_mask_dirs)

        result = {"progress_bar": progress_bar}
        return result

    def visualize(self, batch, warped_cloth, warped_grid):
        # unpack
        im = batch["image"]
        im_cocopose = batch["im_cocopose"]
        maybe_densepose = [batch["densepose"]] if "densepose" in batch else []
        c = batch["cloth"]
        im_h = batch["im_head"]
        silhouette = batch["silhouette"]
        im_c = batch["im_cloth"]
        # layout
        visuals = [
            [im_h, silhouette, im_cocopose] + maybe_densepose,
            [c, warped_cloth, im_c],
            [warped_grid, (warped_cloth + im) * 0.5, im],
        ]
        tensor = tensor_list_for_board(visuals)
        # add to experiment
        for i, img in enumerate(tensor):
            self.logger.experiment.add_image(f"combine/{i:03d}", img, self.global_step)
