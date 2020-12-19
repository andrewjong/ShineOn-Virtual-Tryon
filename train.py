import logging
from typing import Callable

import torch
import os.path as osp
import signal
import sys
import traceback
from argparse import Namespace

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import log
from callbacks import (
    CheckpointCustomFilename,
    SaveOnKeyboardInterrupt,
    CheckpointEveryNSteps,
)
from models import find_model_using_name
from options.test_options import TestOptions
from options.train_options import TrainOptions
from util import str2num

logger = log.setup_custom_logger("logger")

# DDP requires setting the manual seed
# https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#distributed-data-parallel
torch.manual_seed(420)


def main(train=True):
    """ Runs train or test """
    options_obj = TrainOptions() if train else TestOptions()
    opt = options_obj.parse()
    logger.setLevel(getattr(logging, opt.loglevel.upper()))

    model_class = find_model_using_name(opt.model)
    if opt.checkpoint:
        model = model_class.load_from_checkpoint(
            # TODO: we have to manually override all TestOptions for hparams in
            #  __init__, because they're not present in the checkpoint's train options.
            #  We should find a better solution
            opt.checkpoint
        )
        logger.info(f"RESUMED {model_class.__name__} from checkpoint: {opt.checkpoint}")
    else:
        model = model_class(opt)
        logger.info(f"INITIALIZED new {model_class.__name__}")
    model.override_hparams(opt)

    trainer = Trainer(
        resume_from_checkpoint=opt.checkpoint if opt.checkpoint else None,
        **get_hardware_kwargs(opt),
        **get_train_kwargs(opt),
        profiler=True,
    )

    if train:
        save_on_interrupt = make_save_on_interrupt(trainer)
        try:
            trainer.fit(model)
        except Exception as e:
            logger.warning(f"Caught a {type(e)}!")
            logger.error(traceback.format_exc())
            save_on_interrupt(name=e.__class__.__name__)

    else:
        print("Testing........")
        print(opt)
        trainer.test(model)

    logger.info(f"Finished {opt.model}, named {opt.name}!")


def get_hardware_kwargs(opt):
    """ Hardware kwargs for the Trainer """
    hardware_kwargs = vars(
        Namespace(
            gpus=opt.gpu_ids,
            distributed_backend=opt.distributed_backend,
            precision=opt.precision,
        )
    )
    return hardware_kwargs


def get_train_kwargs(opt):
    """
    Return Trainer kwargs specific to training if opt.is_train is True.
    Otherwise return an empty dict.
    """
    if not opt.is_train:
        return {}

    train_kwargs = vars(
        Namespace(
            # Checkpointing
            checkpoint_callback=ModelCheckpoint(save_top_k=5, verbose=True),
            callbacks=[
                # CheckpointCustomFilename(),
                CheckpointEveryNSteps(opt.save_count, prefix=opt.model, verbose=True),
            ],
            default_root_dir=osp.join(opt.experiments_dir, opt.name),
            log_save_interval=opt.display_count,
            # Training and data
            accumulate_grad_batches=opt.accumulated_batches,
            max_epochs=opt.keep_epochs + opt.decay_epochs,
            val_check_interval=str2num(opt.val_check_interval),
            # see https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#replace-sampler-ddp
            replace_sampler_ddp=False,
            limit_train_batches=str2num(opt.limit_train_batches),
            limit_val_batches=str2num(opt.limit_val_batches),
            # Debug
            fast_dev_run=opt.fast_dev_run,
        )
    )
    return train_kwargs


def make_save_on_interrupt(trainer: Trainer) -> Callable:
    """ On interrupt, will save checkpoint """

    def save_on_interrupt(*args, name=""):
        name = f"interrupted_by_{name}" if name else "interrupted_by_Ctrl-C"
        try:
            ckpt_path = osp.join(trainer.checkpoint_callback.dirpath, f"{name}.ckpt")
            logger.warning(
                "Training stopped prematurely. "
                f"Saving Trainer checkpoint to: {ckpt_path}"
            )
            trainer.save_checkpoint(ckpt_path)
        except:
            logger.warning(
                "No checkpoint to save. Either training didn't start, or I'm a "
                "child process."
            )
        exit()

    signal.signal(signal.SIGINT, save_on_interrupt)
    return save_on_interrupt


if __name__ == "__main__":
    main(train=True)
