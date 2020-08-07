import abc
import os.path as osp
import argparse
from typing import Union, List

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import find_dataset_using_name
from datasets.n_frames_interface import maybe_combine_frames_and_channels
import pytorch_lightning as pl


class BaseModel(pl.LightningModule, abc.ABC):
    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
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
        parser.add_argument(
            "--self_attn", action="store_true", help="Add self-attention"
        )
        return parser

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.isTrain = self.hparams.isTrain
        if not self.isTrain:
            ckpt_name = osp.basename(hparams.checkpoint)
            self.test_results_dir = osp.join(
                hparams.result_dir, hparams.name, ckpt_name, hparams.datamode
            )

    def train_dataloader(self) -> DataLoader:
        # create dataset
        dataset = find_dataset_using_name(self.hparams.dataset)(self.hparams)
        # create dataloader
        train_loader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            shuffle=not self.hparams.no_shuffle if self.isTrain else False,
        )
        return train_loader

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # same thing, except for shuffle
        return self.train_dataloader()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda e: 1.0
            - max(0, e - self.hparams.keep_epochs)
            / float(self.hparams.decay_epochs + 1),
        )
        return [optimizer], [scheduler]

    @abc.abstractmethod
    def training_step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def test_step(self, *args, **kwargs):
        pass
