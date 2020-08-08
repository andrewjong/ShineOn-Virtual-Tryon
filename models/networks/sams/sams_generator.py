"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from datasets.tryon_dataset import TryonDataset
from models.networks import BaseNetwork
from .spade import AnySpadeResBlock
from .multispade import MultiSpade
from .attentive_multispade import AttentiveMultiSpade
from .image_encoder import ImageEncoder


class SamsGenerator(BaseNetwork):
    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
        parser = BaseNetwork.modify_commandline_options(parser, is_train)
        parser.add_argument("--norm_G", default="spectralspadesyncbatch3x3")
        parser.add_argument(
            "--num_upsampling_layers",
            choices=("normal", "more", "most"),
            default="normal",
            help="If 'more', adds upsampling layer between the two middle resnet "
            "blocks. If 'most', also add one more upsampling + resnet layer at "
            "the end of the generator",
        )
        parser.add_argument(
            "--encoder_power2_growth",
            type=int,
            default=1,
            help="increase this number to decrease the number of encoding layers",
        )
        parser.add_argument(
            "--encoder_num_same",
            type=int,
            default=3,
            help="decrease this to decrease the number of encoding layers",
        )
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.inputs = hparams.person_inputs + hparams.cloth_inputs
        num_feat = hparams.ngf
        self.out_channels = TryonDataset.RGB_CHANNELS

        # Otherwise, we make the network deterministic by starting with
        # downsampled segmentation map instead of random z

        # parameters according to WC-Vid2Vid page 23
        self.encoder = ImageEncoder(
            self.hparams, hparams.encoder_power2_growth, hparams.encoder_num_same
        )

        label_channels_list = sum(
            getattr(TryonDataset, f"{inp.upper()}_CHANNELS")
            for inp in sorted(self.inputs)
        )
        multispade_class = AttentiveMultiSpade if hparams.self_attn else MultiSpade
        self.head_0 = AnySpadeResBlock(
            16 * num_feat,
            16 * num_feat,
            hparams.norm_G,
            label_channels_list,
            multispade_class,
        )
        self.G_middle_0 = AnySpadeResBlock(
            16 * num_feat,
            16 * num_feat,
            hparams.norm_G,
            label_channels_list,
            multispade_class,
        )
        self.G_middle_1 = AnySpadeResBlock(
            16 * num_feat,
            16 * num_feat,
            hparams.norm_G,
            label_channels_list,
            multispade_class,
        )

        self.up_0 = AnySpadeResBlock(
            16 * num_feat,
            8 * num_feat,
            hparams.norm_G,
            label_channels_list,
            multispade_class,
        )
        self.up_1 = AnySpadeResBlock(
            8 * num_feat,
            4 * num_feat,
            hparams.norm_G,
            label_channels_list,
            multispade_class,
        )
        self.up_2 = AnySpadeResBlock(
            4 * num_feat,
            2 * num_feat,
            hparams.norm_G,
            label_channels_list,
            multispade_class,
        )
        self.up_3 = AnySpadeResBlock(
            2 * num_feat,
            1 * num_feat,
            hparams.norm_G,
            label_channels_list,
            multispade_class,
        )

        final_nc = num_feat

        if hparams.num_upsampling_layers == "most":
            self.up_4 = AnySpadeResBlock(
                1 * num_feat,
                num_feat // 2,
                hparams.norm_G,
                label_channels_list,
                multispade_class,
            )
            final_nc = num_feat // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(
        self,
        prev_synth_outputs: List[Tensor],
        prev_segmaps: List[Tensor],
        current_segmaps_dict: Dict[str, Tensor],
    ):
        """
        Args:
            prev_synth_outputs: previous synthesized frames
            prev_segmaps: segmaps for the previous frames
            current_segmaps_dict: segmentation maps for the current frame

        Returns: synthesized output for the current frame
        """
        prev_synth_outputs = torch.cat(prev_synth_outputs, dim=1)
        prev_segmaps = torch.cat(prev_segmaps, dim=1)
        x = self.encoder(prev_synth_outputs, prev_segmaps)

        x = self.head_0(x, current_segmaps_dict)

        x = self.up(x)
        x = self.G_middle_0(x, current_segmaps_dict)

        if (
            self.hparams.num_upsampling_layers == "more"
            or self.hparams.num_upsampling_layers == "most"
        ):
            x = self.up(x)

        x = self.G_middle_1(x, current_segmaps_dict)

        x = self.up(x)
        x = self.up_0(x, current_segmaps_dict)
        x = self.up(x)
        x = self.up_1(x, current_segmaps_dict)
        x = self.up(x)
        x = self.up_2(x, current_segmaps_dict)
        x = self.up(x)
        x = self.up_3(x, current_segmaps_dict)

        if self.hparams.num_upsampling_layers == "most":
            x = self.up(x)
            x = self.up_4(x, current_segmaps_dict)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x
