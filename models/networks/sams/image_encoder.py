from torch import nn, Tensor

from datasets.tryon_dataset import TryonDataset
from models.networks.sams import SPADE, AnySpadeResBlock


class ImageEncoder(nn.Module):
    """ Andrew's implementation of the image encoder from WC-Vid2Vid """
    def __init__(self, hparams, num_encode_up=5, num_same=3):
        super().__init__()
        assert num_encode_up % 2 == 0, f"Pass a multiple of 2; got {num_encode_up=}"
        total = 16 * hparams.ngf
        start = total // num_encode_up
        step = start
        kwargs = {
            "norm_G": hparams.norm_G,
            "label_channels": TryonDataset.AGNOSTIC_CHANNELS * hparams.n_frames,
            "spade_class": SPADE,
        }
        # comment: what goes in as the segmentation map for the prev outputs encoder?
        #   answer: Arun specifies ONLY the segmentation map, no guidance images.
        #   for us, let's just pass in agnostic
        self.layers = nn.ModuleList(
            [  # the first layer
                AnySpadeResBlock(
                    TryonDataset.RGB_CHANNELS * hparams.n_frames, start, **kwargs
                )
            ]  # up layers
            + [
                AnySpadeResBlock(channels, channels + step, **kwargs)
                for channels in range(start, total, step=step)
            ]  # same layers
            + [AnySpadeResBlock(total, total, **kwargs) for _ in range(num_same)]
        )

    def forward(self, frames: Tensor, maps: Tensor):
        for layer in self.layers:
            x = layer(frames, maps)
        return x

