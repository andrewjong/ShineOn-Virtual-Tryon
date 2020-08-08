import argparse

import pytorch_lightning as pl
from torch.nn import L1Loss

from models import BaseModel
from models.networks.vgg import VGGLoss

""" Self Attentive Multi-Spade """

class SamsModel(BaseModel):
    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser = super(SamsModel, cls).modify_commandline_options(parser, is_train)
        return parser

    def __init__(self):
        self.generator = SamsGenerator() # make this pt lightning

        self.multiscale_discriminator = TODO() # make this pt lightning too?
        self.temporal_discriminator = TODO() # make this pt lightning too?

        self.criterion_l1 = L1Loss()
        self.criterion_vgg = VGGLoss()
        self.crit_adv_multiscale
        self.crit_adv_temporal

    def forward(self, *args, **kwargs):
        self.generator(*args, **kwargs)

    def configure_optimizers(self):
        # must do individual optimizers and schedulers per each network
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0:
            # generator
            outputs = self.generator(batch)

            # calculate loss
            pass
        if optimizer_idx == 1:
            # discriminator, remember to update discriminator slower
            disc_0_outputs = self.multiscale_discriminator(batch)
            disc_1_outputs = self.temporal_discriminator(batch)

            # calculate loss
            pass
