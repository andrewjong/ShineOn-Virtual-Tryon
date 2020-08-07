from typing import Iterable

from torch import Tensor, nn

from models.networks.sams.spade import SPADE


class MultiSpade(SPADE):
    """ N sequential spades for the number of maps, from WC-Vid2Vid page 24 """

    def __init__(self, config_text, norm_nc, label_nc, num_spade: int):
        """
        Duck typing the original Spade class.
        Unclear if WC-Vid2Vid put a duplicate batchnorm layer before the first SPADE
        layer, because technically the first SPADE layer already batchnorms. Leaving
        w/out for now.
        Args:
            num_spade: number of spade layers
            *args: same args as SPADE
        """
        self.spade_layers = nn.ModuleList(
            SPADE(config_text, norm_nc, label_nc) for _ in range(num_spade)
        )

    def forward(self, x: Tensor, segmaps: Iterable[Tensor]):
        """
        Args:
            x: input
            segmaps: an iterable of segmap tensors; maps applied in order

        Returns: transformed x
        """
        assert len(segmaps) == len(self.spade_layers)
        for i, segmap in enumerate(segmaps):
            layer = self.spade_layers[i]
            x = layer(x, segmap)
        return x
