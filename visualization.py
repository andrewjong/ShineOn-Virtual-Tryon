import os

import torch
from PIL import Image


def tensor_for_board(img_tensor):
    assert (
        img_tensor.ndim == 4
    ), f"something's not right, i'm not a standard img_tensor. {img_tensor.shape=}"
    # map into [0,1]
    tensor = (img_tensor.clone() + 1) * 0.5
    try:
        tensor.cpu().clamp(0, 1)
    except:
        tensor.float().cpu().clamp(0, 1)
    if tensor.shape[1] == 1:  # masks, make it RGB
        tensor = tensor.repeat(1, 3, 1, 1)

    return tensor


def tensor_list_for_board(img_tensors_list):
    grid_h = len(img_tensors_list)
    grid_w = max(len(img_tensors) for img_tensors in img_tensors_list)

    batch_size, channel, height, width = tensor_for_board(img_tensors_list[0][0]).size()
    canvas_h = grid_h * height
    canvas_w = grid_w * width
    canvas = torch.FloatTensor(batch_size, channel, canvas_h, canvas_w).fill_(0.5)
    for i, img_tensors in enumerate(img_tensors_list):
        for j, img_tensor in enumerate(img_tensors):
            offset_h = i * height
            offset_w = j * width
            tensor = tensor_for_board(img_tensor)
            canvas[
                :, :, offset_h : offset_h + height, offset_w : offset_w + width
            ].copy_(tensor)
    return canvas


def board_add_image(board, tag_name, img_tensor, step_count):
    tensor = tensor_for_board(img_tensor)

    for i, img in enumerate(tensor):
        board.add_image("%s/%03d" % (tag_name, i), img, step_count)


def board_add_images(board, tag_name, img_tensors_list, step_count):
    tensor = tensor_list_for_board(img_tensors_list)

    for i, img in enumerate(tensor):
        board.add_image(f"{tag_name}/{i:03d}", img, step_count)


def get_save_paths(save_dirs, img_names):
    return [os.path.join(s, i) for s, i in zip(save_dirs, img_names)]


def save_images(img_tensors, img_names, save_dirs):
    """ Save a batch of image tensors """
    if len(save_dirs) == 1:
        save_dirs = [save_dirs] * len(img_names)
    for img_tensor, img_name, save_dir in zip(img_tensors, img_names, save_dirs):
        if "warp-mask" in save_dir and "VitonDataset" not in save_dir:
            # if it's warp mask and we're not VitonDataset, skip saving
            continue
        path = os.path.join(save_dir, img_name)
        if os.path.exists(path):
            # tqdm.write(f"Skipping {path}, already exists!")
            continue

        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        array = tensor.numpy().astype("uint8")
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)
        else:
            raise ValueError(
                "Trying to save an image that is not 1 or 3 channels; "
                f"this is unexpected. {array.shape=}"
            )

        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(array).save(path)
