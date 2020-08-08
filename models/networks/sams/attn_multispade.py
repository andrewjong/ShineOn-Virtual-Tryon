from typing import Iterable

import torch
from torch import Tensor, nn

from models.networks.attention import ATTENTION_TYPES
from models.networks.sams.multispade import MultiSpade
from models.networks.sams.spade import SPADE


class AttentiveMultiSpade(MultiSpade):
    def __init__(
        self, config_text, norm_nc, label_nc, num_spade: int, attn_type="sagan"
    ):
        super().__init__(config_text, norm_nc, label_nc, num_spade)
        norm_class, kernel_size = SPADE.parse_config_text(config_text)

        self.attn_nc = norm_nc * num_spade  # stacked

        attn_class = ATTENTION_TYPES[attn_type]
        self.attention_layer = attn_class(self.attn_nc)
        self.mlp_final = nn.Sequential(
            nn.Conv2d(
                self.attn_nc, norm_nc, kernel_size=kernel_size, padding=kernel_size // 2
            ),
            nn.LeakyReLU()
        )

    def forward(self, x: Tensor, segmaps: Iterable[Tensor]):
        # parallel spade on each segmap
        outputs = [self.spade_layers[i](x, segmap) for i, segmap in enumerate(segmaps)]
        # stack maps on channels
        stacked = torch.cat(outputs, dim=1)
        # attend
        attended = self.attention_layer(stacked)
        # reduce num channels back to og
        result = self.mlp_final(attended)
        return result
