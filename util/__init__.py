import importlib
from collections import Iterable
from typing import List, Tuple, Union

import torch


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace("_", "").lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print(
            f"In {module}, there should be a class whose name matches "
            f"{target_cls_name} in lowercase without underscore(_)"
        )
        exit(0)

    return cls


def str2num(s: str):
    try:
        return int(s)
    except ValueError:
        return float(s)


def multiply(a, b):
    return a * b


def divide(a, b):
    return a / b


def without_key(d, *keys):
    """ Return a dict without specified keys.
    WARNING: modifies the existing dict. Copying is expensive.

    Args:
        d (Dictionary):
    """
    for k in keys:
        d.pop(k)
    return d


def get_prev_data_zero_bounded(data: Union[List, Tuple], end_idx, num_frames):
    start_idx = end_idx - num_frames + 1
    prev_n_data = data[max(0, start_idx) : end_idx]
    if not isinstance(prev_n_data, list) and not isinstance(prev_n_data, tuple):
        prev_n_data = [prev_n_data]
    if start_idx < 0:
        prepend_dupes = [data[0] for _ in range(abs(start_idx))]
        prev_n_data = prepend_dupes + prev_n_data
    return prev_n_data


def get_and_cat_inputs(batch, names):
    inputs = torch.cat([batch[inp] for inp in names], dim=1)
    return inputs
