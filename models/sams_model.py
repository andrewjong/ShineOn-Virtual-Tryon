import argparse
from torch import Tensor
from typing import List, Dict

import pytorch_lightning as pl
import torch
from torch.nn import L1Loss

from models import BaseModel
from models.networks.loss import VGGLoss, GANLoss
from models.networks.sams.sams_generator import SamsGenerator

""" Self Attentive Multi-Spade """


class SamsModel(BaseModel):
    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser = super(SamsModel, cls).modify_commandline_options(parser, is_train)
        parser.set_defaults(inputs=("agnostic", "cloth", "densepose", "flow"))
        parser.add_argument("--norm_G", default="spectralspadesyncbatch3x3")
        parser.add_argument(
            "--gan_mode", default="hinge", choices=GANLoss.AVAILABLE_MODES
        )
        parser.add_argument(
            "--num_upsampling_layers",
            choices=("normal", "more", "most"),
            default="normal",
            help="If 'more', adds upsampling layer between the two middle resnet "
            "blocks. If 'most', also add one more upsampling + resnet layer at "
            "the end of the generator",
        )
        return parser

    def __init__(self, hparams):
        super().__init__(hparams)
        self.generator = SamsGenerator(hparams)

        self.multiscale_discriminator = None  # TODO
        self.temporal_discriminator = None  # TODO

        self.criterion_gan = GANLoss("hinge")
        self.criterion_l1 = L1Loss()
        self.criterion_vgg = VGGLoss()
        self.crit_adv_multiscale = None  # TODO
        self.crit_adv_temporal = None  # TODO

    def forward(self, *args, **kwargs):
        self.generator(*args, **kwargs)

    def configure_optimizers(self):
        # must do individual optimizers and schedulers per each network
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0:
            result = self._generator_step(batch, batch_idx)
        else:
            result = self._discriminator_step(batch, batch)
            pass
            # discriminator, remember to update discriminator slower
            # disc_0_outputs = self.multiscale_discriminator(batch)
            # disc_1_outputs = self.temporal_discriminator(batch)

        return result

    def _generator_step(self, batch, batch_idx):
        # List of: [ agnostic_frames, densepose_frames, flow_frames ].
        #  each is of length "n_frames"
        segmaps: Dict[str : List[Tensor]] = {
            key: batch[key] for key in self.hparams.inputs
        }
        # make a buffer of previous frames
        shape = segmaps["image"][0]
        prev_frames: List[Tensor] = [torch.zeros(*shape) for _ in range(self.n_frames)]
        # generate previous frames before this one
        for frame_idx in range(self.n_frames):
            # prepare data
            this_frame_segmaps: Dict[str:Tensor] = {
                key: segmap[frame_idx] for key, segmap in segmaps.items()
            }
            # forward
            synth_output: Tensor = self.generator.forward(
                prev_frames, TODO, this_frame_segmaps
            )
            # comment: should we detach()? Ziwei says yes
            prev_frames[frame_idx] = synth_output.detach()

        # loss
        ground_truth = batch["image"][-1]
        loss_gan = self.criterion_gan()
        loss_l1 = self.criterion_l1(synth_output, ground_truth)
        loss_vgg = self.criterion_vgg(synth_output, ground_truth)
        # TODO: ADVERSARIAL LOSSES

        loss = loss_gan + loss_l1 + loss_vgg

        result = {"loss": loss}
        return result

    def _discriminator_step(self, batch, batch_idx, optimizer_idx):
        pass
