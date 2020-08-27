"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
from typing import Type, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision

from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
from models.networks.activation import Sine, Swish


class SPADE(nn.Module):
    """
    Creates SPADE normalization layer based on the given configuration
    SPADE consists of two steps. First, it normalizes the activations using
    your favorite normalization method, such as Batch Norm or Instance Norm.
    Second, it applies scale and bias to the normalized output, conditioned on
    the segmentation map.
    The format of |config_text| is spade(norm)(ks), where
    (norm) specifies the type of parameter-free normalization.
          (e.g. syncbatch, batch, instance)
    (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
    Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
    Also, the other arguments are
    |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
    |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
    """

    @staticmethod
    def parse_config_text(config_text: str) -> Tuple[Type[nn.Module], int]:
        """
        Args:
            config_text: something like spadeinstance3x3

        Returns: norm class, kernel size
        """
        assert config_text.startswith("spade")
        parsed = re.search("spade(\D+)(\d)x\d", config_text)
        param_free_norm_type = str(parsed.group(1))
        if param_free_norm_type == "instance":
            param_free_norm = nn.InstanceNorm2d
        elif param_free_norm_type == "syncbatch":
            param_free_norm = SynchronizedBatchNorm2d
        elif param_free_norm_type == "batch":
            param_free_norm = nn.BatchNorm2d
        else:
            raise ValueError(
                "%s is not a recognized param-free norm type in SPADE"
                % param_free_norm_type
            )
        ks = int(parsed.group(2))
        return param_free_norm, ks

    def __init__(self, config_text, norm_nc, label_nc, activation):
        super().__init__()

        param_free_norm_cls, ks = SPADE.parse_config_text(config_text)
        self.param_free_norm = param_free_norm_cls(norm_nc, affine=False)
        self.actvn = self._get_activation_fn(activation)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        self.nhidden = nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), self.actvn
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return Swish()
        elif activation == "sine":
            return Sine()
        else:
            raise RuntimeError(f"The selected activation should be relu/gelu/swish/sine, not {activation}")


class AnySpadeResBlock(nn.Module):
    """
    Modified from original SPADEResnetBlock, accepts a spade_class argument

    -- Original Text --
    It differs from the ResNet block of pix2pixHD in that
    it takes in the segmentation map as input, learns the skip connection if necessary,
    and applies normalization first and then convolution.
    This architecture seemed like a standard architecture for unconditional or
    class-conditional GAN architecture using residual block.
    The code was inspired from https://github.com/LMescheder/GAN_stability.
    """

    def __init__(
        self,
        fin: int,
        fout: int,
        norm_G: str,
        label_channels: Union[int, Dict[str,int]],
        spade_class: Type[SPADE],
        activation: str
    ):
        """

        Args:
            fin:
            fout:
            norm_G:
            label_channels:
            spade_class: SPADE, MultiSpade, or AttnMultiSpade
        """
        super().__init__()
        # Attributes
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if "spectral" in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm_G.replace("spectral", "")
        self.spade_0 = spade_class(spade_config_str, fin, label_channels, activation)
        self.spade_1 = spade_class(spade_config_str, fmiddle, label_channels, activation)
        if self.learned_shortcut:
            self.norm_s = spade_class(spade_config_str, fin, label_channels, activation)
        self.actvn = self._get_activation_fn(activation)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.spade_0(x, seg)))
        dx = self.conv_1(self.actvn(self.spade_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return nn.LeakyReLU(2e-1)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return Swish()
        elif activation == "sine":
            return Sine()
        else:
            raise RuntimeError(f"The selected activation should be relu/gelu/swish/sine, not {activation}")


class ResnetBlock(nn.Module):
    """
    ResNet block used in pix2pixHD
    We keep the same architecture as pix2pixHD.
    """

    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out
