import functools
from abc import ABC, abstractmethod
from argparse import ArgumentParser

import torch
from torch.utils.data.dataloader import default_collate


class NFramesInterface(ABC):
    """
    Given an index, collect N frames
    Adds the --n_frames to commandline args.

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
            "--n_frames",
            type=int,
            default=1,
            help="Total number of frames to load at once. This is useful for video "
            "training. Set to 1 for images.",
        )
        return parser

    def __init__(self, opt):
        """ sets n_frames """
        self._n_frames = opt.n_frames
        assert self._n_frames >= 1, "--n_frames Must be a positive integer"

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
            assert len(prev_indices) == self._n_frames, (
                f"{len(prev_indices) = } doesn't match {self._n_frames = }, "
                f"something's wrong!"
            )

            frames = [getitem_func(self, i) for i in prev_indices]

            collated = default_collate(frames)
            return collated

        return wrapper


def maybe_combine_frames_and_channels(opt, inputs):
    """ if n_frames is true, combines frames and channels dim for all the tensors"""
    if hasattr(opt, "n_frames"):

        def maybe_combine(t):
            if isinstance(t, torch.Tensor):
                bs, n_frames, c, h, w = t.shape
                t = t.view(bs, n_frames * c, h, w)
            return t

        inputs = {k: maybe_combine(v) for k, v in inputs.items()}

    return inputs
