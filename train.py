import logging
import os.path as osp
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import log
from models import find_model_using_name
from options.test_options import TestOptions
from options.train_options import TrainOptions

logger = log.setup_custom_logger("logger")


def main(train=True):
    options_obj = TrainOptions() if train else TestOptions()
    opt = options_obj.parse()
    logger.setLevel(getattr(logging, opt.loglevel.upper()))

    model_class = find_model_using_name(opt.model)
    model = model_class(opt)
    if opt.checkpoint or not train:
        checkpoint = torch.load(opt.checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)# = model_class.load_from_checkpoint(opt.checkpoint)

    root_dir = osp.join(opt.experiments_dir, opt.name)


    trainer = Trainer(
        gpus=opt.gpu_ids,
        default_root_dir=root_dir,
        max_steps=opt.datacap,
        log_save_interval=opt.display_count,
        fast_dev_run=opt.fast_dev_run,
        max_epochs=opt.keep_epochs + opt.decay_epochs,
        val_check_interval=opt.save_count,
        resume_from_checkpoint=opt.checkpoint if opt.checkpoint else None
    )
    if train:
        trainer.fit(model)
    else:
        trainer.test(model)

    logger.info(f"Finished {opt.model}, named {opt.name}!")


if __name__ == "__main__":
    main(train=True)
