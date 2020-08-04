import argparse
import os

import torch

import datasets


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument("--name", default="GMM")
        # compute
        parser.add_argument(
            "--gpu_ids", default="0", help="comma separated of which GPUs to train on"
        )
        parser.add_argument("-j", "--workers", type=int, default=4)
        parser.add_argument("-b", "--batch_size", type=int, default=8)
        # data
        parser.add_argument(
            "--dataset", choices=("viton", "viton_vvt_mpv", "vvt", "mpv"), default="vvt"
        )
        parser.add_argument("--datamode", default="train")
        parser.add_argument("--stage", default="GMM")
        parser.add_argument(
            "--dataparallel",
            action="store_true",
            help="pass flag to enable dataparallel model",
        )
        parser.add_argument(
            "--datacap",
            type=float,
            default=float("inf"),
            help="limits the dataset to this many batches",
        )
        # network dimensions
        parser.add_argument(
            "--person_in_channels",
            type=int,
            default=1 + 3 + 18,  # silhouette + head + cocopose
            help="number of base channels for person representation",
        )
        parser.add_argument(
            "--cloth_in_channels",
            type=int,
            default=3,
            help="number of channels for cloth representation",
        )
        parser.add_argument(
            "--densepose",
            action="store_true",
            help="use me to add densepose (auto adds 3 to --person_in_channels)",
        )
        parser.add_argument("--fine_width", type=int, default=192)
        parser.add_argument("--fine_height", type=int, default=256)
        parser.add_argument("--radius", type=int, default=5)
        parser.add_argument("--grid_size", type=int, default=5)
        # logging
        parser.add_argument(
            "--tensorboard_dir",
            type=str,
            default="tensorboard",
            help="save tensorboard infos. pass empty string '' to disable tensorboard",
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="checkpoints",
            help="save checkpoint infos",
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            default="",
            help="model checkpoint for initialization",
        )
        parser.add_argument(
            "--display_count",
            type=int,
            help="how often to update tensorboard, in steps",
            default=100,
        )
        parser.add_argument(
            "--loglevel",
            choices=("debug", "info", "warning", "error", "critical"),
            default="info",
            help="choose a log level",
        )
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        # model_name = opt.model
        # model_option_setter = models.get_option_setter(model_name)
        # parser = model_option_setter(parser, self.isTrain)
        # opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset
        dataset_option_setter = datasets.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, "{}_opt.txt".format(opt.datamode))
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

        self.options_formatted_str = message

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # # process opt.suffix
        # if opt.suffix:
        #     suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
        #     opt.name = opt.name + suffix
        #

        opt = BaseOptions.apply_gpu_ids(opt)
        opt = BaseOptions.apply_densepose_to_person_in_channels(opt)

        self.print_options(opt)

        self.opt = opt
        return self.opt

    @staticmethod
    def apply_gpu_ids(opt):
        # set gpu ids
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        print(opt.gpu_ids)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        return opt

    @staticmethod
    def apply_densepose_to_person_in_channels(opt):
        if opt.densepose:
            opt.person_in_channels += 3
        return opt
