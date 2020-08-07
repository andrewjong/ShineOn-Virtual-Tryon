import argparse

import pytorch_lightning as pl

from models import BaseModel

""" Self Attentive Multi-Spade """

class SamsModel(BaseModel):
    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser = super(SamsModel, cls).modify_commandline_options(parser, is_train)
        return parser

    def __init__(self):
        pass

    def forward(self):
        pass


    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0:
            # generator
            pass
        if optimizer_idx == 1:
            # discriminator
            pass
    pass