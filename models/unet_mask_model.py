import argparse
import math
import os.path as osp

import torch
from torch import nn as nn
from torch.nn import functional as F

from datasets.n_frames_interface import maybe_combine_frames_and_channels
from models.base_model import BaseModel, get_and_cat_inputs
from models.networks.loss import VGGLoss
from models.networks.cpvton.unet import UnetGenerator
from visualization import tensor_list_for_board, save_images, get_save_paths


class UnetMaskModel(BaseModel):
    """ CP-VTON Try-On Module (TOM) """

    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser = super(UnetMaskModel, cls).modify_commandline_options(parser, is_train)
        parser.set_defaults(
            person_inputs=("agnostic", "densepose"), cloth_inputs=("cloth")
        )
        return parser

    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        n_frames = hparams.n_frames if hasattr(hparams, "n_frames") else 1
        self.unet = UnetGenerator(
            input_nc=(self.person_channels + self.cloth_channels) * n_frames,
            output_nc=4 * n_frames,
            num_downs=6,
            # scale up the generator features conservatively for the number of images
            ngf=int(64 * (math.log(n_frames) + 1)),
            norm_layer=nn.InstanceNorm2d,
            use_self_attn=hparams.self_attn,
        )
        self.criterionVGG = VGGLoss()

    def forward(self, person_representation, warped_cloths):
        # comment andrew: Do we need to interleave the concatenation? Or can we leave it
        #  like this? Theoretically the unet will learn where things are, so let's try
        #  simple concat for now.
        concat_tensor = torch.cat([person_representation, warped_cloths], 1)
        outputs = self.unet(concat_tensor)

        # teach the u-net to make the 1st part the rendered images, and
        # the 2nd part the masks
        boundary = 4 * self.hparams.n_frames - self.hparams.n_frames
        p_rendereds = outputs[:, 0:boundary, :, :]
        m_composites = outputs[:, boundary:, :, :]

        p_rendereds = F.tanh(p_rendereds)
        m_composites = F.sigmoid(m_composites)

        # chunk for operation per individual frame
        warped_cloths_chunked = torch.chunk(warped_cloths, self.hparams.n_frames)
        p_rendereds_chunked = torch.chunk(p_rendereds, self.hparams.n_frames)
        m_composites_chunked = torch.chunk(m_composites, self.hparams.n_frames)

        p_tryons = [
            wc * mask + p * (1 - mask)
            for wc, p, mask in zip(
                warped_cloths_chunked, p_rendereds_chunked, m_composites_chunked
            )
        ]
        p_tryons = torch.cat(p_tryons, dim=1)  # cat back to the channel dim

        return p_rendereds, m_composites, p_tryons

    def training_step(self, batch, batch_idx):
        batch = maybe_combine_frames_and_channels(self.hparams, batch)
        # unpack
        im = batch["image"]
        cm = batch["cloth_mask"]
        person_inputs = get_and_cat_inputs(batch, self.hparams.person_inputs)
        cloth_inputs = get_and_cat_inputs(batch, self.hparams.cloth_inputs)

        # forward
        p_rendered, m_composite, p_tryon = self.forward(person_inputs, cloth_inputs)
        # loss
        loss_image_l1 = F.l1_loss(p_tryon, im)
        loss_image_vgg = self.criterionVGG(p_tryon, im)
        loss_mask_l1 = F.l1_loss(m_composite, cm)
        loss = loss_image_l1 + loss_image_vgg + loss_mask_l1

        # logging
        if self.global_step % self.hparams.display_count == 0:
            self.visualize(batch, p_rendered, m_composite, p_tryon)

        progress_bar = {
            "loss_image_l1": loss_image_l1,
            "loss_image_vgg": loss_image_vgg,
            "loss_mask_l1": loss_mask_l1,
        }
        tensorboard_scalars = {
            "epoch": self.current_epoch,
            "loss": loss,
        }
        tensorboard_scalars.update(progress_bar)
        result = {
            "loss": loss,
            "log": tensorboard_scalars,
            "progress_bar": progress_bar,
        }
        return result

    def test_step(self, batch, batch_idx):
        dataset_names = batch["dataset_name"]
        # use subfolders for each subdataset
        try_on_dirs = [
            osp.join(self.test_results_dir, dname, "try-on") for dname in dataset_names
        ]

        im_names = batch["im_name"]
        # if we already did a forward-pass on this batch, skip it
        save_paths = get_save_paths(im_names, try_on_dirs)
        if all(osp.exists(s) for s in save_paths):
            progress_bar = {"file": f"Skipping {im_names[0]}"}
        else:
            progress_bar = {"file": f"{im_names[0]}"}

            person_inputs = get_and_cat_inputs(batch, self.hparams.person_inputs)
            cloth_inputs = get_and_cat_inputs(batch, self.hparams.cloth_inputs)

            p_rendered, m_composite, p_tryon = self.forward(person_inputs, cloth_inputs)

            save_images(p_tryon, im_names, try_on_dirs)

        result = {"progress_bar": progress_bar}
        return result

    def visualize(self, b, p_rendered, m_composite, p_tryon):
        # unpack
        maybe_densepose = [b["densepose"]] if "densepose" in b else []
        # layout
        visuals = [
            [b["im_head"], b["silhouette"], b["im_cocopose"]] + maybe_densepose,
            [b["cloth"], b["cloth_mask"] * 2 - 1, m_composite * 2 - 1],
            [p_rendered, p_tryon, b["image"]],
        ]
        tensor = tensor_list_for_board(visuals)
        # add to experiment
        for i, img in enumerate(tensor):
            self.logger.experiment.add_image(f"combine/{i:03d}", img, self.global_step)
