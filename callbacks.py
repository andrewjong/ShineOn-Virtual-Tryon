import os.path as osp
import pytorch_lightning as pl


class CheckpointCustomFilename(pl.Callback):
    def __init__(self, filename_fmt="{epoch}_{global_step}_{val_loss:.2f}"):
        """
        Formats just the filename of the Trainer checkpoints, such that the path
        organization is preserved and only the filename format is modified.

        Filename format can include any desired metrics.

        Args:
            filename_fmt: see formats here, excluding the preceeding path https://pytorch-lightning.readthedocs.io/en/latest/callbacks.html#model-checkpointing .
              (default: "{epoch}_{global_step}_{val_loss:.2f}")
        """
        self.filename_fmt = filename_fmt

    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        """
        ModelCheckpoint hardcodes self.filename = '{epoch}' in its on_train_start().
        But custom callbacks are called _before_ ModelCheckpoint, meaning setting it
        in our on_train_start() would just get overwritten. Therefore, we set it here in
        on_validation_end(), as checkpointing in Lightning is currently tied to
        Validation performance.
        """
        trainer.checkpoint_callback.filename = self.filename_fmt

    def on_validation_start(self, trainer: pl.Trainer, pl_module):
        """
        Add start too just in case
        """
        trainer.checkpoint_callback.filename = self.filename_fmt

    def on_train_start(self, trainer: pl.Trainer, pl_module):
        """
        Add start too just in case
        """
        trainer.checkpoint_callback.filename = self.filename_fmt

    def on_train_end(self, trainer: pl.Trainer, pl_module):
        """
        Add start too just in case
        """
        trainer.checkpoint_callback.filename = self.filename_fmt


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """

        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
              use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
              filename, don't use ours.
        """
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix

    def on_batch_end(self, trainer: pl.Trainer, _):
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = osp.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


class SaveOnKeyboardInterrupt(pl.Callback):
    def on_keyboard_interrupt(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        ckpt_path = osp.join(
            trainer.checkpoint_callback.dirpath, "SaveOnKeyboardInterruptCallback.ckpt"
        )
        trainer.save_checkpoint(ckpt_path)
