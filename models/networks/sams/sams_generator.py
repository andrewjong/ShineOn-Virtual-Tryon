"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from datasets.tryon_dataset import TryonDataset
from models.networks.sams.attn_multispade import AttentiveMultiSpade
from models.networks.sams.multispade import MultiSpade
from models.networks.sams.spade import AnySpadeResBlock, SPADE


class SamsGenerator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        num_feat = hparams.ngf
        self.out_channels = TryonDataset.RGB_CHANNELS

        # Otherwise, we make the network deterministic by starting with
        # downsampled segmentation map instead of random z

        # parameters according to WC-Vid2Vid page 23
        self.encoder = self.define_encoder(num_encode_up=5, num_same=3)

        label_channels_list = sum(
            getattr(TryonDataset, f"{inp.upper()}_CHANNELS")
            for inp in sorted(hparams.inputs)
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

    def define_encoder(self, num_encode_up, num_same):
        """
        Creates the encoder for the previous N frames
        Args:
            num_encode_up: number of layers to sample up to the total amount (16 * self.hparams.ngf)
            num_same: number of layers to keep the channels the same

        Returns: encoder layers
        """
        assert num_encode_up % 2 == 0, f"Pass a multiple of 2; got {num_encode_up=}"
        total = 16 * self.hparams.ngf
        start = total // num_encode_up
        step = start
        kwargs = {
            "norm_G": self.hparams.norm_G,
            "label_channels_list": TryonDataset.RGB_CHANNELS,
            "spade_class": SPADE,
        }
        # TODO: what goes in as the segmentation map for the prev outputs encoder?
        layers = [
            AnySpadeResBlock(
                TryonDataset.RGB_CHANNELS * self.hparams.n_frames, start, **kwargs
            )
        ]
        layers.extend(
            [
                AnySpadeResBlock(channels, channels + step, **kwargs)
                for channels in range(start, total, step=step)
            ]
        )
        layers.extend(
            [AnySpadeResBlock(total, total, **kwargs) for _ in range(num_same)]
        )
        encoder = nn.Sequential(*layers)
        return encoder

    def forward(self, prev_synth_outputs: List[Tensor], segmaps_list: List[Tensor]):
        """
        Args:
            prev_synth_outputs: previous synthesized frames
            segmaps_list: segmentation maps for the current frame

        Returns: synthesized output for the current frame
        """
        # TODO: what goes in as the segmentation map for the prev outputs encoder?
        prev_synth_outputs = torch.cat(prev_synth_outputs, dim=1)
        x = self.encoder(prev_synth_outputs)

        x = self.head_0(x, segmaps_list)

        x = self.up(x)
        x = self.G_middle_0(x, segmaps_list)

        if (
            self.hparams.num_upsampling_layers == "more"
            or self.hparams.num_upsampling_layers == "most"
        ):
            x = self.up(x)

        x = self.G_middle_1(x, segmaps_list)

        x = self.up(x)
        x = self.up_0(x, segmaps_list)
        x = self.up(x)
        x = self.up_1(x, segmaps_list)
        x = self.up(x)
        x = self.up_2(x, segmaps_list)
        x = self.up(x)
        x = self.up_3(x, segmaps_list)

        if self.hparams.num_upsampling_layers == "most":
            x = self.up(x)
            x = self.up_4(x, segmaps_list)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x
