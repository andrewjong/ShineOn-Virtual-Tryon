import logging

import torch
import argparse
import sys

import datasets
import models

logger = logging.getLogger("logger")


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument("--name", default="unnamed_experiment")
        # compute
        parser.add_argument(
            "--gpu_ids", default="0", help="comma separated of which GPUs to train on"
        )
        parser.add_argument("-j", "--workers", type=int, default=4)
        parser.add_argument("-b", "--batch_size", type=int, default=8)
        parser.add_argument(
            "-fp",
            "--precision",
            type=int,
            dest="precision",
            help="choose fp16 (half) or fp32 (full) precision training",
            choices=(16, 32),
            default=16,
        )
        # data
        parser.add_argument(
            "--dataset", choices=("viton", "viton_vvt_mpv", "vvt", "mpv"), default="vvt"
        )
        parser.add_argument("--datamode", default="train")
        parser.add_argument(
            "--model",
            help="which model to use. choices: "
            "'warp' (aka 'gmm'), 'unet_mask' (aka 'tom'), 'sams'.",
        )
        parser.add_argument(
            "--datacap",
            type=float,
            default=float("inf"),
            help="limits the DataLoader to this many batches",
        )
        # logging
        parser.add_argument(
            "--experiments_dir",
            default="experiments",
            help="where to store logs and checkpoints",
        )
        # parser.add_argument(
        #     "--tensorboard_dir",
        #     type=str,
        #     default="tensorboard",
        #     help="save tensorboard infos. pass empty string '' to disable tensorboard",
        # )
        # parser.add_argument(
        #     "--checkpoint_dir",
        #     type=str,
        #     default="checkpoints",
        #     help="save checkpoint infos",
        # )
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
            default=200,
        )
        parser.add_argument(
            "--loglevel",
            choices=("debug", "info", "warning", "error", "critical"),
            default="info",
            help="choose a log level",
        )
        # debug
        parser.add_argument(
            "--fast_dev_run", action="store_true", help="quickly test out the pipeline",
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

        opt = BaseOptions.apply_model_synonyms(opt)
        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

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
        # expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
        # os.makedirs(expr_dir, exist_ok=True)
        # file_name = os.path.join(expr_dir, "{}_opt.txt".format(opt.datamode))
        # with open(file_name, "wt") as opt_file:
        #     opt_file.write(message)
        #     opt_file.write("\n")

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

        opt = BaseOptions.apply_ask_unnamed_experiment(opt)
        opt = BaseOptions.apply_model_synonyms(opt)
        opt = BaseOptions.apply_gpu_ids(opt)
        opt = BaseOptions.apply_half_precision_if_pytorch_1_6(opt)
        opt = BaseOptions.apply_val_check_interval_number_type(opt)
        opt = BaseOptions.apply_sort_inputs(opt)
        from datasets.n_frames_interface import NFramesInterface

        opt = NFramesInterface.apply_n_frames_now_default_total(opt)
        from models.sams_model import SamsModel

        opt = SamsModel.apply_default_encoder_input(opt)

        self.print_options(opt)

        self.opt = opt
        return self.opt

    @staticmethod
    def apply_ask_unnamed_experiment(opt):
        if "--name" not in sys.argv:
            print(
                "\n"
                "You didn't set an experiment name. Do you want to set one? If not, "
                f"leave it blank. This message can be avoided by passing --name NAME."
            )
            new_name = input(f"Experiment name (default: {opt.name}): ")
            print()
            if new_name:
                opt.name = new_name
                print(f"Experiment name set to {opt.name}")
        return opt

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
        return opt

    @staticmethod
    def apply_model_synonyms(opt):
        opt.model = opt.model.lower()
        before = opt.model
        if opt.model == "gmm":
            opt.model = "warp"
        elif opt.model == "tom" or opt.model == "unet":
            opt.model = "unet_mask"

        if before != opt.model:
            print(f"User passed --model {before}, assuming you meant {opt.model}")
        return opt

    @staticmethod
    def apply_sort_inputs(opt):
        opt.person_inputs = sorted(opt.person_inputs)
        opt.cloth_inputs = sorted(opt.cloth_inputs)
        return opt

    @staticmethod
    def apply_val_check_interval_number_type(opt):
        if hasattr(opt, "val_check_interval"):
            if "." in opt.val_check_interval:
                opt.val_check_interval = float(opt.val_check_interval)
            else:
                opt.val_check_interval = int(opt.val_check_interval)
                if opt.datacap < opt.val_check_interval:
                    opt.val_check_interval = int(opt.datacap)
        return opt

    @staticmethod
    def apply_half_precision_if_pytorch_1_6(opt):
        if torch.__version__ < "1.6.0" and opt.precision == 16:
            logger.warning(
                "Cannot use half precision with PyTorch < 1.6.0. "
                f"Detected {torch.__version__ = }. Changing to full precision (fp32)."
            )
            opt.precision = 32
        return opt
