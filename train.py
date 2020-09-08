import logging
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


def main(train=True):
    options_obj = TrainOptions() if train else TestOptions()
    opt = options_obj.parse()
    logger.setLevel(getattr(logging, opt.loglevel.upper()))

    model_class = find_model_using_name(opt.model)

    hardware_kwargs = vars(Namespace(
        gpus=opt.gpu_ids,
        distributed_backend=opt.distributed_backend,
        precision=opt.precision,
    ))

    if opt.checkpoint or not train:
        model = model_class.load_from_checkpoint(opt.checkpoint)
        trainer = Trainer(
            resume_from_checkpoint=opt.checkpoint,
            # Hardware
            **hardware_kwargs,
        )
    else:
        model = model_class(opt)
        trainer = Trainer(
            # Hardware
            **hardware_kwargs,
            # Checkpointing
            checkpoint_callback=ModelCheckpoint(save_top_k=5, verbose=True),
            callbacks=[
                CheckpointCustomFilename(),
                CheckpointEveryNSteps(opt.save_count, verbose=True),
            ],
            default_root_dir=osp.join(opt.experiments_dir, opt.name),
            log_save_interval=opt.display_count,
            # Training and data
            limit_train_batches=str2num(opt.limit_train_batches),
            limit_val_batches=str2num(opt.limit_val_batches),
            max_epochs=opt.keep_epochs + opt.decay_epochs,
            val_check_interval=str2num(opt.val_check_interval),
            # Debug
            fast_dev_run=opt.fast_dev_run,
            accumulate_grad_batches=opt.accumulated_batches,
            profiler=True,
        )

    if train:

        def save_on_interrupt(*args, name=""):
            name = f"interrupted_by_{name}" if name else "interrupted_by_Ctrl-C"
            try:
                ckpt_path = osp.join(
                    trainer.checkpoint_callback.dirpath, f"{name}.ckpt"
                )
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
        try:
            trainer.fit(model)
        except Exception as e:
            logger.warning(f"Caught a {type(e)}!")
            logger.error(traceback.format_exc())
            save_on_interrupt(name=e.__class__.__name__)
    else:
        print("testing........")
        print(opt)
        trainer.test(model)

    logger.info(f"Finished {opt.model}, named {opt.name}!")


if __name__ == "__main__":
    main(train=True)
