import abc
import argparse
import logging
import os.path as osp
from pprint import pformat
from typing import List, Dict
from torch import Tensor

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter

from datasets import find_dataset_using_name
from datasets.tryon_dataset import TryonDataset, parse_num_channels
from datasets.vvt_dataset import VVTDataset
from datasets.n_frames_interface import maybe_combine_frames_and_channels
logger = logging.getLogger("logger")


class BaseModel(pl.LightningModule, abc.ABC):
    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
        # network dimensions
        parser.add_argument(
            "--person_inputs",
            nargs="+",
            help="List of what type of items are passed as person input. Dynamically"
            "sets input tensors and number of channels. See TryonDataset for "
            "options.",
        )
        parser.add_argument(
            "--cloth_inputs",
            nargs="+",
            default=("cloth",),
            help="List of items to pass as the cloth inputs.",
        )
        parser.add_argument("--ngf", type=int, default=64)
        parser.add_argument(
            "--self_attn", action="store_true", help="Add self-attention"
        )
        parser.add_argument(
            "--no_self_attn",
            action="store_false",
            dest="self_attn",
            help="No self-attention",
        )
        parser.add_argument(
            "--flow_warp", action="store_true", help="Warp the previous frame with flow"
        )
        return parser

    def __init__(self, hparams, *args, **kwargs):
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.n_frames_total = hparams.n_frames_total

        self.person_channels = parse_num_channels(hparams.person_inputs)
        self.cloth_channels = parse_num_channels(hparams.cloth_inputs)

        self.isTrain = self.hparams.isTrain
        if not self.isTrain:
            ckpt_name = osp.basename(hparams.checkpoint)
            self.test_results_dir = osp.join(
                hparams.result_dir, hparams.name, ckpt_name, hparams.datamode
            )

    def prepare_data(self) -> None:
        # hacky, log hparams to tensorboard; lightning currently has problems with
        # this: https://github.com/PyTorchLightning/pytorch-lightning/issues/1228
        board: SummaryWriter = self.logger.experiment
        board.add_text("hparams", pformat(self.hparams, indent=4, width=1))

    def setup(self, stage):
        dataset_cls = find_dataset_using_name(self.hparams.dataset)
        self.train_dataset: TryonDataset = dataset_cls(self.hparams)
        logger.info(
            f"Train {self.hparams.dataset} dataset initialized: "
            f"{len(self.train_dataset)} samples."
        )
        if stage == "fit":
            self.val_dataset = self.train_dataset.make_validation_dataset(self.hparams)
            logger.info(
                f"Val {self.hparams.dataset} dataset initialized: "
                f"{len(self.val_dataset)} samples."
            )

    def train_dataloader(self) -> DataLoader:
        # create dataloader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=not self.hparams.no_shuffle,
            num_workers=self.hparams.workers
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        # create dataloader
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=not self.hparams.no_shuffle,
            num_workers=self.hparams.workers
        )
        return val_loader

    def validation_step(self, batch, idx):
        """ Must set self.batch = batch for validation_end() to visualize the last
        sample"""
        self.batch = batch
        result = self.training_step(batch, idx, val=True)
        return result

    def visualize(self, input_batch, tag="train"):
        """ Any outputs to visualize should be saved to self """
        pass

    def on_validation_epoch_end(self) -> None:
        if hasattr(self, "batch"):
            self.visualize(self.batch, "validation")

    def test_dataloader(self) -> DataLoader:
        # same loader type. test paths will be defined in hparams
        return self.train_dataloader()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.hparams.lr)
        scheduler = self._make_step_scheduler(optimizer)
        return [optimizer], [scheduler]

    def _make_step_scheduler(self, optimizer):
        def step_func(epoch):
            decrease = max(0, epoch - self.hparams.keep_epochs) / float(
                self.hparams.decay_epochs + 1
            )
            decay = 1.0 - decrease
            if decay < 1.0:
                logger.info(
                    f"{epoch=}, multiplied original learning rate by {decay:.2f}"
                )

            return decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=step_func)
        return scheduler

    def fetch_person_visuals(self, batch, sort_fn=None) -> List[torch.Tensor]:
        """
        Gets the correct tensors for --person_inputs. Can sort it with sort_fn if
        desired.
        Args:
            batch:
            sort_fn: function to sort in desired order; function should return List[str]
        """
        person_vis_names = self.replace_actual_with_visual()
        if sort_fn:
            person_vis_names = sort_fn(person_vis_names)
        person_visual_tensors = []
        for name in person_vis_names:
            tensor: torch.Tensor = batch[name]
            channels = tensor.shape[-3]

            if channels <= VVTDataset.RGB_CHANNELS:
                person_visual_tensors.append(tensor)
            else:
                logger.warning(
                    f"Tried to visualize a tensor > {VVTDataset.RGB_CHANNELS} channels:"
                    f" '{name}' tensor has {channels=}, {tensor.shape=}. Skipping it."
                )
        if len(person_visual_tensors) == 0:
            raise ValueError("Didn't find any tensors to visualize!")

        return person_visual_tensors

    def replace_actual_with_visual(self) -> List[str]:
        """
        Replaces non-RGB names with the names of their visualizations.
        Returns a list copy.
        """
        person_visuals: List[str] = self.hparams.person_inputs.copy()
        if "agnostic" in person_visuals:
            i = person_visuals.index("agnostic")
            person_visuals.pop(i)
            person_visuals.insert(i, "im_head")
            person_visuals.insert(i, "silhouette")

        if "cocopose" in person_visuals:
            i = person_visuals.index("cocopose")
            person_visuals.pop(i)
            person_visuals.insert(i, "im_cocopose")

        if "flow" in person_visuals:
            i = person_visuals.index("flow")
            person_visuals.pop(i)
            if self.hparams.visualize_flow:
                person_visuals.insert(i, "flow_image")

        return person_visuals
