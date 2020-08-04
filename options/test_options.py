from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            "--result_dir", type=str, default="result", help="save result infos"
        )
        # parser.add_argument(...)
        self.isTrain = False
        return parser
