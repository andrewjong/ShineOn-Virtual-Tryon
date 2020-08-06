""" Also known as GMM """
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from datasets.n_frames_interface import maybe_combine_frames_and_channels
from models.networks.cpvton import (
    FeatureExtraction,
    FeatureL2Norm,
    FeatureCorrelation,
    FeatureRegression,
    TpsGridGen,
)
from visualization import tensor_list_for_board


# coding=utf-8


class GMM(pl.LightningModule):
    """ Geometric Matching Module
    """

    @staticmethod
    def modify_commandline_options(parser: ArgumentParser, is_train):
        parser = ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument("--ngf", type=int, default=12)
        parser.add_argument("--grid_size", type=int, default=5)
        return parser

    def __init__(self, opt):
        self.hparams = opt
        self.opt = opt
        super(GMM, self).__init__()
        # n_frames = opt.n_frames if hasattr(opt, "n_frames") else 1
        self.extractionA = FeatureExtraction(
            opt.person_in_channels,  # 1 + 3 + 18 + 3
            ngf=64,
            n_layers=3,
            norm_layer=nn.BatchNorm2d,
        )
        self.extractionB = FeatureExtraction(
            3, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d
        )
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(
            input_nc=192, output_dim=2 * opt.grid_size ** 2
        )
        self.gridGen = TpsGridGen(
            opt.fine_height, opt.fine_width, grid_size=opt.grid_size
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

    def on_batch_start(self, batch):
        batch = maybe_combine_frames_and_channels(self.opt, batch)
        return batch

    def training_step(self, batch):

        agnostic = batch["agnostic"]
        c = batch["cloth"]
        im_c = batch["im_cloth"]
        im_g = batch["grid_vis"]

        # forward
        grid, theta = self.forward(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode="border")
        warped_grid = F.grid_sample(im_g, grid, padding_mode="zeros")

        loss = F.l1_loss(warped_cloth, im_c)

        # Logging
        if self.global_step % self.opt.display_count == 0:
            self.visualize(batch, warped_cloth, warped_grid)
            board = self.logger.experiment
            board.add_scalar("epoch", self.current_epoch, self.global_step)
            board.add_scalar("metric", loss.item(), self.global_step)

        result = pl.TrainResult()
        result.log("l1 loss", loss, prog_bar=True)

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
        visuals = [
            [im_h, silhouette, im_cocopose] + maybe_densepose,
            [c, warped_cloth, im_c],
            [warped_grid, (warped_cloth + im) * 0.5, im],
        ]
        tensor = tensor_list_for_board(visuals)
        for i, img in enumerate(tensor):
            self.logger.experiment.add_image(f"combine/{i:03d}", img, self.global_step)
