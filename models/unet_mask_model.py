import argparse
import logging
import math
import os.path as osp
from typing import List

import torch
from pytorch_lightning import TrainResult, EvalResult
from torch import nn as nn
from torch.nn import functional as F
from torchvision.utils import save_image

from datasets.n_frames_interface import maybe_combine_frames_and_channels
from datasets.tryon_dataset import TryonDataset
from datasets.vvt_dataset import VVTDataset
from models.base_model import BaseModel
from models.networks import init_weights
from models.networks.cpvton.unet import UnetGenerator
from models.networks.loss import VGGLoss
from util import get_and_cat_inputs
from visualization import tensor_list_for_board, save_images, get_save_paths
from .flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d

logger = logging.getLogger("logger")


class UnetMaskModel(BaseModel):
    """ CP-VTON Try-On Module (TOM) """

    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser = super(UnetMaskModel, cls).modify_commandline_options(parser, is_train)
        parser.set_defaults(person_inputs=("agnostic", "densepose"))
        parser.add_argument(
            "--pen_flow_mask",
            type=float,
            default=1.0,
            help="Penalty applied to flow mask loss",
        )
        return parser

    def __init__(self, hparams):
        super().__init__(hparams)
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        n_frames = hparams.n_frames_total if hasattr(hparams, "n_frames_total") else 1
        self.unet = UnetGenerator(
            input_nc=(self.person_channels + self.cloth_channels) * n_frames,
            output_nc=5 * n_frames if self.hparams.flow_warp else 4 * n_frames,
            num_downs=6,
            num_attention=hparams.num_attn if hasattr(hparams, "num_attn") else 2,
            # scale up the generator features conservatively for the number of images
            ngf=int(64 * (math.log(n_frames) + 1)),
            norm_layer=nn.InstanceNorm2d,
            use_self_attn=hparams.self_attn,
            activation=hparams.activation,
        )
        self.resample = Resample2d()
        self.criterionVGG = VGGLoss()
        init_weights(self.unet, init_type="normal")

    def forward(self, person_representation, warped_cloths, flows=None, prev_im=None):
        # comment andrew: Do we need to interleave the concatenation? Or can we leave it
        #  like this? Theoretically the unet will learn where things are, so let's try
        #  simple concat for now.

        concat_tensor = torch.cat([person_representation, warped_cloths], 1)
        outputs = self.unet(concat_tensor)

        # teach the u-net to make the 1st part the rendered images, and
        # the 2nd part the masks
        boundary = 3 * self.hparams.n_frames_total
        weight_boundary = 4 * self.hparams.n_frames_total

        p_rendereds = outputs[:, 0:boundary, :, :]
        tryon_masks = outputs[:, boundary:weight_boundary, :, :]

        flow_masks = (
            outputs[:, weight_boundary:, :, :] if self.hparams.flow_warp else None
        )

        p_rendereds = F.tanh(p_rendereds)
        tryon_masks = F.sigmoid(tryon_masks)
        flow_masks = F.sigmoid(flow_masks) if flow_masks is not None else None

        # chunk operation per individual frame
        flows = (
            list(torch.chunk(flows, self.hparams.n_frames_total, dim=1))
            if flows is not None
            else None
        )
        warped_cloths_chunked = list(
            torch.chunk(warped_cloths, self.hparams.n_frames_total, dim=1)
        )
        p_rendereds_chunked = list(
            torch.chunk(p_rendereds, self.hparams.n_frames_total, dim=1)
        )
        tryon_masks_chunked = list(
            torch.chunk(tryon_masks, self.hparams.n_frames_total, dim=1)
        )
        flow_masks_chunked = (
            list(torch.chunk(flow_masks, self.hparams.n_frames_total, dim=1))
            if flow_masks is not None
            else None
        )

        # only use second frame for warping
        all_generated_frames = []
        for fIdx in range(self.hparams.n_frames_total):
            if flows is not None and fIdx > 0:
                # Warp previous frame to current frame using flow
                prev_generated_frame = all_generated_frames[fIdx - 1]
                warped_by_flow = self.resample(
                    prev_generated_frame, flows[fIdx].contiguous()
                )
                p_rendered_X_warped_by_flow = (
                    (1 - flow_masks_chunked[fIdx]) * warped_by_flow
                    + flow_masks_chunked[fIdx] * p_rendereds_chunked[fIdx]
                )
                p_rendered = p_rendered_X_warped_by_flow
            else:
                p_rendered = p_rendereds_chunked[fIdx]

            p_tryon = (
                (1 - tryon_masks_chunked[fIdx]) * p_rendered  ##
                + tryon_masks_chunked[fIdx] * warped_cloths_chunked[fIdx]
            )

            all_generated_frames.append(p_tryon)

        p_tryons = torch.cat(all_generated_frames, dim=1)  # cat back to the channel dim

        return p_rendereds, tryon_masks, p_tryons, flow_masks

    def training_step(self, batch, batch_idx, val=False):
        batch = maybe_combine_frames_and_channels(self.hparams, batch)
        # unpack
        im = batch["image"]
        prev_im = batch["prev_image"]
        cm = batch["cloth_mask"]
        flow = batch["flow"] if self.hparams.flow_warp else None

        person_inputs = get_and_cat_inputs(batch, self.hparams.person_inputs)
        cloth_inputs = get_and_cat_inputs(batch, self.hparams.cloth_inputs)

        # forward. save outputs to self for visualization
        (
            self.p_rendereds,
            self.tryon_masks,
            self.p_tryons,
            self.flow_masks,
        ) = self.forward(person_inputs, cloth_inputs, flow, prev_im)
        self.p_tryons = torch.chunk(self.p_tryons, self.hparams.n_frames_total, dim=1)
        self.p_rendereds = torch.chunk(
            self.p_rendereds, self.hparams.n_frames_total, dim=1
        )
        self.tryon_masks = torch.chunk(
            self.tryon_masks, self.hparams.n_frames_total, dim=1
        )


        self.flow_masks = (
            torch.chunk(self.flow_masks, self.hparams.n_frames_total, dim=1)
            if self.flow_masks is not None
            else None
        )

        im = torch.chunk(im, self.hparams.n_frames_total, dim=1)
        cm = torch.chunk(cm, self.hparams.n_frames_total, dim=1)

        # loss
        loss_image_l1_curr = F.l1_loss(self.p_tryons[-1], im[-1])
        loss_image_l1_prev = F.l1_loss(self.p_tryons[-2], im[-2]) if self.hparams.n_frames_total > 1 else torch.zeros_like(loss_image_l1_curr)
        loss_image_l1 = 0.5 * (loss_image_l1_curr + loss_image_l1_prev) if self.hparams.n_frames_total > 1 else loss_image_l1_curr

        loss_image_vgg_curr = self.criterionVGG(self.p_tryons[-1], im[-1])
        loss_image_vgg_prev = self.criterionVGG(self.p_tryons[-2], im[-2]) if self.hparams.n_frames_total > 1 else torch.zeros_like(loss_image_vgg_curr)
        loss_image_vgg = 0.5 * (loss_image_vgg_curr + loss_image_vgg_prev) if self.hparams.n_frames_total > 1 else loss_image_vgg_curr

        loss_tryon_mask_curr = F.l1_loss(self.tryon_masks[-1], cm[-1])
        loss_tryon_mask_prev = F.l1_loss(self.tryon_masks[-2], cm[-2]) if self.hparams.n_frames_total > 1 else torch.zeros_like(loss_tryon_mask_curr)
        loss_tryon_mask_l1 = 0.5 * (loss_tryon_mask_curr + loss_tryon_mask_prev) if self.hparams.n_frames_total > 1 else loss_tryon_mask_curr

        loss_flow_mask_l1 = (
            self.flow_masks[-1].sum() if self.flow_masks is not None else torch.zeros_like(loss_tryon_mask_curr)
        ) * self.hparams.pen_flow_mask

        loss = loss_image_l1 + loss_image_vgg + loss_tryon_mask_l1 + loss_flow_mask_l1

        # logging
        if not val and self.global_step % self.hparams.display_count == 0:
            self.visualize(batch)

        val_ = "val_" if val else ""
        result = EvalResult(checkpoint_on=loss) if val else TrainResult(loss)
        result.log(f"{val_}loss/G", loss, prog_bar=True)

        result.log(f"{val_}loss/G/l1", loss_image_l1, prog_bar=True)
        result.log(f"{val_}loss/G/vgg", loss_image_vgg, prog_bar=True)
        result.log(f"{val_}loss/G/tryon_mask_l1", loss_tryon_mask_l1, prog_bar=True)
        result.log(f"{val_}loss/G/flow_mask_l1", loss_flow_mask_l1, prog_bar=True)


        if self.hparams.n_frames_total > 1:
            ## visualize prev frames losses
            result.log(f"{val_}loss/G/l1_prev", loss_image_l1_prev)
            result.log(f"{val_}loss/G/vgg_prev", loss_image_vgg_prev)
            result.log(f"{val_}loss/G/tryon_mask_prev", loss_tryon_mask_prev)

            ## visualize curr frames losses
            result.log(f"{val_}loss/G/l1_curr", loss_image_l1_curr)
            result.log(f"{val_}loss/G/vgg_curr", loss_image_vgg_curr)
            result.log(f"{val_}loss/G/tryon_mask_curr", loss_tryon_mask_curr)
        #from IPython import embed; embed()
        return result


    def visualize(self, b, tag="train"):
        if tag == "validation":
            b = maybe_combine_frames_and_channels(self.hparams, b)
        person_visuals = self.fetch_person_visuals(b)
        visuals = [
            person_visuals,
            [
                # extract only the latest frame (for --n_frames_total)
                b["cloth"][:, -TryonDataset.CLOTH_CHANNELS :, :, :],
                b["cloth_mask"][:, -TryonDataset.CLOTH_MASK_CHANNELS :, :, :] * 2 - 1,
                self.tryon_masks[-TryonDataset.MASK_CHANNELS] * 2 - 1,
            ],
            [
                self.p_rendereds[-1],
                self.p_tryons[-1],
                b["image"][:, -TryonDataset.RGB_CHANNELS :, :, :],
                b["prev_image"][:, -TryonDataset.RGB_CHANNELS :, :, :],
            ],
        ]
        for list_i in range(len(visuals)):
            for list_j in range(len(visuals[list_i])):
                tensor = visuals[list_i][list_j]
                if tensor.dim() == 5:
                    tensor = torch.squeeze(tensor, 1)
                    visuals[list_i][list_j] = tensor
        tensor = tensor_list_for_board(visuals)
        # add to experiment
        for i, img in enumerate(tensor):
            self.logger.experiment.add_image(f"{tag}/{i:03d}", img, self.global_step)

    def test_step(self, batch, batch_idx):
        batch = maybe_combine_frames_and_channels(self.hparams, batch)
        dataset_names = batch["dataset_name"]
        im_names = batch["image_name"]
        if self.hparams.n_frames_total > 1:
            dataset_names = get_last_item_per_batch(dataset_names)
            im_names = get_last_item_per_batch(im_names)

        task = "tryon" if self.hparams.tryon_list else "reconstruction"
        try_on_dirs = [
            osp.join(self.test_results_dir, dname, task) for dname in dataset_names
        ]

        # if we already did a forward-pass on this batch, skip it
        save_paths = get_save_paths(try_on_dirs, im_names)
        if all(osp.exists(s) for s in save_paths):
            progress_bar = {"file": f"Skipping {im_names[0]}"}
        else:
            progress_bar = {"file": f"{im_names[0]}"}

            person_inputs = get_and_cat_inputs(batch, self.hparams.person_inputs)
            cloth_inputs = get_and_cat_inputs(batch, self.hparams.cloth_inputs)

            _, _, self.p_tryon, _ = self.forward(person_inputs, cloth_inputs)

            # TODO CLEANUP: we get the last frame here by picking the last RGB channels;
            #  this is different from how it's done in training_step, which uses
            #  chunking and -1 indexing. We should choose one method for consistency.
            save_images(
                self.p_tryon[:, -TryonDataset.RGB_CHANNELS :, :,], im_names, try_on_dirs
            )
        result = {"progress_bar": progress_bar}
        return result

    def fetch_person_visuals(self, batch, sort_fn=None) -> List[torch.Tensor]:
        """
        Gets the correct tensors for --person_inputs. Can sort it with sort_fn if
        desired.
        Args:
            batch:
            sort_fn: function to sort in desired order; function should return List[str]
        """
        person_vis_names = self.replace_actual_with_visual()
        if sort_fn:
            person_vis_names = sort_fn(person_vis_names)
        person_visual_tensors = []
        for name in person_vis_names:
            tensor: torch.Tensor = batch[name]
            if self.hparams.n_frames_total > 1:
                channels = tensor.shape[-3] // self.hparams.n_frames_total
                tensor = tensor[:, -1 * channels :, :, :]
            else:
                channels = tensor.shape[-3]

            if (
                channels == VVTDataset.RGB_CHANNELS
                or channels == VVTDataset.CLOTH_MASK_CHANNELS
            ):
                person_visual_tensors.append(tensor)
            else:
                logger.warning(
                    f"Tried to visualize a tensor > {VVTDataset.RGB_CHANNELS} channels:"
                    f" '{name}' tensor has {channels=}, {tensor.shape=}. Skipping it."
                )
        if len(person_visual_tensors) == 0:
            raise ValueError("Didn't find any tensors to visualize!")

        return person_visual_tensors


def get_last_item_per_batch(*args):
    ret_tuple = []
    for item_batch in args:
        just_latest = [seq[-1] for seq in item_batch]
        ret_tuple.append(just_latest)

    return ret_tuple if len(ret_tuple) > 1 else ret_tuple[0]
