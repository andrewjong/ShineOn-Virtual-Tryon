from typing import Iterable

import torch
from torch import Tensor

from models.networks.attention import sagan, ATTENTION_TYPES
from models.networks.sams.multispade import MultiSpade


class AttentiveMultiSpade(MultiSpade):
    def __init__(
        self, config_text, norm_nc, label_nc, num_spade: int, attn_type="sagan"
    ):
        super().__init__(config_text, norm_nc, label_nc, num_spade)

        self.attn_nc = norm_nc * num_spade  # stacked

        attn_class = ATTENTION_TYPES[attn_type]
        self.attention_layer = attn_class(self.attn_nc)

    def forward(self, x: Tensor, segmaps: Iterable[Tensor]):
        # parallel computation
        outputs = [self.spade_layers[i](x, segmap) for i, segmap in enumerate(segmaps)]
        stacked = torch.cat(outputs, dim=1)  # stack on channels
        attended = self.attention_layer(stacked)
        # TODO: do we need to reduce the number of channels back to the original now?
        return attended

