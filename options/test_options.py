from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.set_defaults(no_shuffle=True)
        parser.set_defaults(datamode="test")
        self.is_train = False
        parser.add_argument(
            "--result_dir",
            type=str,
            default="test_results",
            help="save test result outputs",
        )

        parser.add_argument(
            "--tryon_list",
            help="Use a CSV file to specify what cloth should go on each person."
            "The CSV should have two columns: CLOTH_PATH and PERSON_ID. "
            "Cloth_path is the path to the image of the cloth product to wear. "
            "Person_id is the identifier that corresponds to the ID under each "
            "annotation folder.",
        )
        parser.add_argument(
            "--random_tryon",
            help="Randomly choose cloth-person pairs for try-on. ",
            action="store_true",
        )
        # parser.add_argument(...)

        return parser
