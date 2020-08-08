from typing import Iterable, List

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

    def forward(self, x: Tensor, segmaps_list: List[Tensor]):
        """
        Args:
            x: input
            segmaps_list: list of segmaps

        Returns: transformed x
        """
        if isinstance(segmaps_list, Tensor):
            segmaps_list = [segmaps_list]
        # split on channel dimensions
        assert len(segmaps_list) == len(self.spade_layers)
        for i, segmap in enumerate(segmaps_list):
            layer = self.spade_layers[i]
            x = layer(x, segmap)
        return x
