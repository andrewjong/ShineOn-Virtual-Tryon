from typing import Iterable, List, Dict

import torch
from torch import Tensor, nn

from models.networks.attention import ATTENTION_TYPES
from models.networks.sams import MultiSpade
from models.networks.sams.spade import SPADE


class AttentiveMultiSpade(MultiSpade):
    def __init__(
        self,
        config_text: str,
        norm_nc: int,
        label_channels_dict: Dict[str:int],
        attn_type: str = "sagan",
    ):
        super().__init__(config_text, norm_nc, label_channels_dict)
        norm_class, kernel_size = SPADE.parse_config_text(config_text)

        num_spade = len(self.spade_layers)
        self.attn_nc = norm_nc * num_spade  # stacked

        attn_class = ATTENTION_TYPES[attn_type]
        self.attention_layer = attn_class(self.attn_nc)
        self.mlp_final = nn.Sequential(
            nn.Conv2d(
                self.attn_nc, norm_nc, kernel_size=kernel_size, padding=kernel_size // 2
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor, segmaps_dict: Dict[str:Tensor]):
        if isinstance(segmaps_dict, Tensor):
            segmaps_dict = {MultiSpade.DEFAULT_KEY: segmaps_dict}
        # parallel spade on each segmap
        outputs = [
            self.spade_layers[key](x, segmap)
            for key, segmap in self.sort_fn(segmaps_dict.items())
            # ^ ordering matters for the below concatenation
        ]
        # stack maps on channels
        outputs_together = torch.cat(outputs, dim=1)
        # attend
        attended = self.attention_layer(outputs_together)
        # reduce num channels back to og
        result = self.mlp_final(attended)
        return result
