from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.set_defaults(no_shuffle=True)
        self.isTrain = False

        # parser.add_argument(...)

        return parser
