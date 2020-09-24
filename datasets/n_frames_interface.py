import collections
import functools
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Dict

import torch
from torch.utils.data.dataloader import default_collate



class NFramesInterface(ABC):
    """
    Given an index, collect N frames
    Adds the --n_frames_total to commandline args.

    Usage:
    ```
    class VideoDataset(NFramesInterface):
        @staticmethod
        def modify_commandline_options(parser: argparse.ArgumentParser, is_train):
            parser = NFramesInterface.modify_commandline_options(parser, is_train)
            ...
            return parser

        @NFramesInterface.return_n_frames
        def __getitem__(self, index):
            # ... code to return 1 item will be duplicated for the indices defined in
            collect_n_frames_indices
    ```
    """

    @staticmethod
    def modify_commandline_options(parser: ArgumentParser, is_train):
        parser.add_argument(
            "--n_frames_total",
            type=int,
            default=1,
            metavar="N",
            help="Total number of frames to load at once. This is useful for video "
            "training. Set to 1 for images.",
        )
        parser.add_argument(
            "--n_frames_now",
            type=int,
            default=None,
            metavar="N",
            help="Use progressive video training by slowly incrementing "
            "--n_frames_now from 1 up to --n_frames_total. Any frames at an index "
            "between --n_frames_now and --n_frames_total will simply be filled with"
            " zeros. Setting to None disables progressive training.",
        )
        return parser

    @staticmethod
    def apply_n_frames_now_default_total(opt):
        """ Call in Base Options after opt parsed """
        if opt.n_frames_now is None:
            opt.n_frames_now = opt.n_frames_total
        return opt

    def __init__(self, opt):
        """ sets n_frames_total """
        self.n_frames_total = opt.n_frames_total
        self.n_frames_now = opt.n_frames_now
        assert self.n_frames_total >= 1, "--n_frames_total Must be a positive integer"
        assert (
            self.n_frames_now <= self.n_frames_total
        ), f"{opt.n_frames_now} > {opt.n_frames_total}"

    @abstractmethod
    def collect_n_frames_indices(self, index):
        """ Define the indices to return, given the current index"""
        pass

    @staticmethod
    def return_n_frames(getitem_func):
        """ Decorator to get n frames based on the subclass's implementation of
        `collect_n_frames_indices()`. Should be used to decorate `__getitem__`.

        Args:
            getitem_func (function): the `__getitem__` function"""

        @functools.wraps(getitem_func)
        def wrapper(self, index):
            if not isinstance(self, NFramesInterface):
                raise ValueError(
                    "Can only use this decorator in subclasses of PrevFramesDataset"
                )
            # use the subclass's implementation
            prev_indices = self.collect_n_frames_indices(index)
            assert len(prev_indices) == self.n_frames_total, (
                f"{len(prev_indices) = } doesn't match {self.n_frames_total = }, "
                f"something's wrong!"
            )

            frames = [getitem_func(self, i) for i in prev_indices]

            collated = default_collate(frames)
            return collated

        return wrapper


def maybe_combine_frames_and_channels(opt, inputs: Dict, has_batch_dim=True):
    """
    if n_frames_total is true, combines frames and channels dim for all the tensors.
    For tuples, unpacks it from the list that wraps it.

    Args:
        opt:
        inputs:
        has_batch_dim: whether or not batch dim is already included. If called within
            a dataset class, this should be False. If called as the output of a
            dataloader, then should be True.
    """
    if hasattr(opt, "n_frames_total"):

        def maybe_combine(t):
            # Tensor like items
            if isinstance(t, torch.Tensor):
                if has_batch_dim and len(t.shape) == 5:
                    bs, n_frames, c, h, w = t.shape
                    t = t.view(bs, n_frames * c, h, w)
                elif not has_batch_dim and len(t.shape) == 4:
                    n_frames, c, h, w = t.shape
                    t = t.view(n_frames * c, h, w)
            # Non-tensor like items, such as lists of strings or numbers
            elif isinstance(t, collections.abc.Sequence) and not isinstance(t, str):
                # unpack
                if opt.n_frames_total == 1:
                    t = t[0]

            return t

        new_inputs = {k: maybe_combine(v) for k, v in inputs.items()}

    return new_inputs
