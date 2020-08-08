from math import log2

from torch import nn, Tensor

from datasets.tryon_dataset import TryonDataset
from models.networks.sams import SPADE, AnySpadeResBlock


class ImageEncoder(nn.Module):
    """ Andrew's implementation of the image encoder from WC-Vid2Vid """

    def __init__(self, hparams, power2_growth=1, num_same=3):
        super().__init__()
        total_features = 16 * hparams.ngf
        kwargs = {
            "norm_G": hparams.norm_G,
            "label_channels": TryonDataset.AGNOSTIC_CHANNELS * hparams.n_frames,
            "spade_class": SPADE,
        }
        # comment: what goes in as the segmentation map for the prev outputs encoder?
        #   answer: Arun specifies ONLY the segmentation map, no guidance images.
        #   for us, let's just pass in agnostic
        start_step = 5
        end_step = int(log2(total_features))
        assert end_step >= start_step, f"{end_step=} !>= {start_step=}"
        self.layers = nn.ModuleList(
            [  # the first layer
                AnySpadeResBlock(
                    TryonDataset.RGB_CHANNELS * hparams.n_frames, 32, **kwargs
                )
            ]  # up layers
            + [
                AnySpadeResBlock(2 ** power, 2 ** (power + 1), **kwargs)
                for power in range(start_step, end_step, power2_growth)
            ]  # same layers
            + [
                AnySpadeResBlock(total_features, total_features, **kwargs)
                for _ in range(num_same)
            ]
        )

    def forward(self, frames: Tensor, maps: Tensor):
        x = frames
        for layer in self.layers:
            x = layer(x, maps)
        return x
