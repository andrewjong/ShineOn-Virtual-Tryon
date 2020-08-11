import argparse

from models.networks import GANLoss


def modify_commandline_options(parser: argparse.ArgumentParser, is_train):
    if is_train:
        parser.add_argument(
            "--gan_mode", default="hinge", choices=GANLoss.AVAILABLE_MODES
        )
        parser.add_argument(
            "--lr_D",
            type=float,
            default=3e-4,
            help="Learning rate for Discriminators. Recommend setting to "
                 "x2 or x4 of --lr (generator's) according to the TTUR rule "
                 "(Heusel et al. 2017)",
        )
        parser.add_argument(
            "--no_ganFeat_loss",
            action="store_true",
            help="Disable GAN feature matching in loss. Not recommended; "
                 "Feature matching improves GAN stability (Salimans et al. 2016)",
        )
    return parser
