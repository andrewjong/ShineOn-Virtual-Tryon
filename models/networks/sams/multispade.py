from typing import Iterable

import torch
from torch import Tensor, nn

from models.networks.sams.spade import SPADE


class MultiSpade(SPADE):
    """ N sequential spades for the number of maps, from WC-Vid2Vid page 24 """

    def __init__(self, config_text, norm_nc, label_channels_list):
        """
        Duck typing the original Spade class.
        Unclear if WC-Vid2Vid put a duplicate batchnorm layer before the first SPADE
        layer, because technically the first SPADE layer already batchnorms. Leaving
        w/out for now.
        Args:
            num_spade: number of spade layers
            norm_nc: number of channels through norm
            label_channels_list: number of channels for the segmentation maps
        """
        # purposefully avoid super call(), we are ducktyping

        if isinstance(label_channels_list, int):
            label_channels_list = [label_channels_list]
        self.label_channels_list = label_channels_list
        self.spade_layers = nn.ModuleList(
            SPADE(config_text, norm_nc, label_nc)
            for label_nc in label_channels_list
        )

    def forward(self, x: Tensor, segs_concatted: Tensor):
        """
        Args:
            x: input
            segs_concatted: segmaps concatenated on channel in the correct order

        Returns: transformed x
        """
        # split on channel dimensions
        segmaps_list = torch.split(segs_concatted, self.label_channels_list, dim=1)
        assert len(segmaps_list) == len(self.spade_layers)
        for i, segmap in enumerate(segmaps_list):
            layer = self.spade_layers[i]
            x = layer(x, segmap)
        return x
