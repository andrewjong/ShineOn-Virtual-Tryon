""" Also known as GMM """
import argparse
import os.path as osp
from argparse import ArgumentParser
from typing import List

import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import EvalResult, TrainResult

from datasets.n_frames_interface import maybe_combine_frames_and_channels
from models.base_model import BaseModel
from util import get_and_cat_inputs
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
        parser.add_argument("--grid_size", type=int, default=5)
        parser.set_defaults(person_inputs=("agnostic", "cocopose"))
        # TODO: We don't have densepose created for VITON and MPV yet
        # parser.set_defaults(person_inputs=("agnostic", "densepose"))
        return parser

    def __init__(self, hparams):
        super(WarpModel, self).__init__(hparams)
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        # n_frames_total = opt.n_frames_total if hasattr(opt, "n_frames_total") else 1
        self.extractionA = FeatureExtraction(
            self.person_channels,
            ngf=hparams.ngf,
            n_layers=3,
            norm_layer=nn.BatchNorm2d,
        )
        self.extractionB = FeatureExtraction(
            self.cloth_channels, ngf=hparams.ngf, n_layers=3, norm_layer=nn.BatchNorm2d
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

    def training_step(self, batch, idx, val=False):
        batch = maybe_combine_frames_and_channels(self.hparams, batch)
        # unpack
        c = batch["cloth"]
        im_c = batch["im_cloth"]
        im_g = batch["grid_vis"]
        person_inputs = get_and_cat_inputs(batch, self.hparams.person_inputs)
        cloth_inputs = get_and_cat_inputs(batch, self.hparams.cloth_inputs)

        # forward
        grid, theta = self.forward(person_inputs, cloth_inputs)
        self.warped_cloth = F.grid_sample(c, grid, padding_mode="border")
        self.warped_grid = F.grid_sample(im_g, grid, padding_mode="zeros")
        # loss
        loss = F.l1_loss(self.warped_cloth, im_c)

        # Logging
        if not val and self.global_step % self.hparams.display_count == 0:
            self.visualize(batch)

        val_ = "val_" if val else ""
        result = EvalResult(checkpoint_on=loss) if val else TrainResult(loss)
        result.log(f"{val_}loss/G", loss, prog_bar=True)

        return result

    def visualize(self, b, tag="train"):
        if tag == "validation":
            b = maybe_combine_frames_and_channels(self.hparams, b)
        person_visuals = self.fetch_person_visuals(b)

        visuals = [
            person_visuals,
            [b["cloth"], self.warped_cloth, b["im_cloth"]],
            [self.warped_grid, (self.warped_cloth + b["image"]) * 0.5, b["image"]],
        ]
        tensor = tensor_list_for_board(visuals)
        # add to experiment
        for i, img in enumerate(tensor):
            self.logger.experiment.add_image(f"{tag}/{i:03d}", img, self.global_step)

    def test_step(self, batch, batch_idx):
        batch = maybe_combine_frames_and_channels(self.hparams, batch)
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
        save_paths = get_save_paths(warp_cloth_dirs, c_names)
        if all(osp.exists(s) for s in save_paths):
            progress_bar = {"file": f"Skipping {c_names[0]}"}
        else:
            progress_bar = {"file": c_names[0]}
            # unpack the the data
            c = batch["cloth"]
            cm = batch["cloth_mask"]
            im_g = batch["grid_vis"]
            person_inputs = get_and_cat_inputs(batch, self.hparams.person_inputs)
            cloth_inputs = get_and_cat_inputs(batch, self.hparams.cloth_inputs)

            # forward pass
            grid, theta = self.forward(person_inputs, cloth_inputs)
            self.warped_cloth = F.grid_sample(c, grid, padding_mode="border")
            warped_mask = F.grid_sample(cm, grid, padding_mode="zeros")
            self.warped_grid = F.grid_sample(im_g, grid, padding_mode="zeros")

            # save images
            save_images(self.warped_cloth, c_names, warp_cloth_dirs)
            save_images(warped_mask * 2 - 1, c_names, warp_mask_dirs)

        result = {"progress_bar": progress_bar}
        return result
