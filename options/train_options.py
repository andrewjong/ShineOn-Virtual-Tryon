import argparse

from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser: argparse.ArgumentParser):
        parser = BaseOptions.initialize(self, parser)
        # data
        parser.add_argument("--no_shuffle", action="store_true", help="don't shuffle input data")
        # checkpoints

        parser.add_argument(
            "--val_check_interval",
            "--val_frequency",
            dest="val_check_interval",
            type=float,
            default=0.125,
            help="If float, validate (and checkpoint) after this many epochs. "
                 "If int, validate after this many batches. If 0 or 0.0, validate "
                 "every step."
        )
        # optimization
        parser.add_argument(
            "--lr", type=float, default=1e-4, help="initial learning rate for adam"
        )

        self.isTrain = True
        return parser
