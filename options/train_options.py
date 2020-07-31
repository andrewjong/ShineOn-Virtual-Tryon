from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            "--save_count",
            type=int,
            help="how often to save a checkpoint, in epochs",
            default=1,
        )
        parser.add_argument(
            "--keep_epochs",
            type=int,
            help="number of epochs with initial learning rate",
            default=100,
        )
        parser.add_argument(
            "--decay_epochs",
            type=int,
            help="number of epochs to linearly decay the learning rate",
            default=100,
        )
        self.isTrain = True
        return parser
