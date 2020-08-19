import argparse
import torch
from torch import nn
from models.networks.sams.sams_generator import SamsGenerator

hparams = argparse.Namespace(
    norm_G="spectralspadesyncbatch3x3", 
    encoder_input="RGB", 
    person_inputs=["densepose"],
    cloth_inputs=[],
    ngf=1, 
    n_frames=1,
    ngf_base=2,
    ngf_pow_outer=6,
    ngf_pow_inner=10,
    ngf_pow_step=4,
    num_middle=3,
    self_attn=False
)

gen = SamsGenerator(hparams)

x = torch.randn(1, 3, 256, 192)
densepose = torch.randn(1, 3, 256, 192)

out = gen(x, densepose, densepose)