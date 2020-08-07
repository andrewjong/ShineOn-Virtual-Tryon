import pytorch_lightning as pl
""" Self Attentive Multi-Spade """

class SamsModel(pl.LightningModule):
    @staticmethod
    def modify_commandline_options(parser, is_train):

        parser.add_argument("--no_gan_loss", dest="gan_loss", action="store_false")
        parser.add_argument("--no_attention", dest="attention", action="store_false", help=("disable attention"))
        parser.add_argument("--attention_location", choices=("combine-spade", "post-layer"))
        pass

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