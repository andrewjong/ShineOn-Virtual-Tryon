# coding=utf-8
import logging
import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import find_dataset_using_name
import log
from datasets.n_frames_interface import maybe_combine_frames_and_channels
from models.networks.cpvton import load_checkpoint
from models.unet_masking_model import TOM
from models.warp_model import GMM
from options.test_options import TestOptions
from visualization import board_add_images, save_images, get_save_paths

logger = log.setup_custom_logger("logger")

def test_gmm(opt, test_loader, model, board):
    device = torch.device("cuda", opt.gpu_ids[0])
    model.to(device)
    model.eval()

    base_name = os.path.basename(opt.checkpoint)

    save_root = os.path.join(opt.result_dir, opt.name, base_name, opt.datamode)
    print(f"Saving to {save_root}")

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, inputs in pbar:
        inputs = maybe_combine_frames_and_channels(opt, inputs)
        dataset_names = inputs["dataset_name"]
        # produce subfolders for each subdataset
        warp_cloth_dirs = [
            osp.join(save_root, dname, "warp-cloth") for dname in dataset_names
        ]
        warp_mask_dirs = [
            osp.join(save_root, dname, "warp-mask") for dname in dataset_names
        ]

        c_names = inputs["cloth_name"]
        # if we already did a forward-pass on this batch, skip it
        save_paths = get_save_paths(c_names, warp_cloth_dirs)
        if all(os.path.exists(s) for s in save_paths):
            pbar.set_description(f"Skipping {c_names[0]}")
            continue

        pbar.set_description(c_names[0])
        # unpack the rest of the data
        im = inputs["image"].to(device)
        im_cocopose = inputs["im_cocopose"].to(device)
        maybe_densepose = (
            [inputs["densepose"].to(device)] if "densepose" in inputs else []
        )
        im_h = inputs["im_head"].to(device)
        silhouette = inputs["silhouette"].to(device)
        agnostic = inputs["agnostic"].to(device)
        c = inputs["cloth"].to(device)
        cm = inputs["cloth_mask"].to(device)
        im_c = inputs["im_cloth"].to(device)
        im_g = inputs["grid_vis"].to(device)

        # forward pass
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode="border")
        warped_mask = F.grid_sample(cm, grid, padding_mode="zeros")
        warped_grid = F.grid_sample(im_g, grid, padding_mode="zeros")

        # save images
        save_images(warped_cloth, c_names, warp_cloth_dirs)
        save_images(warped_mask * 2 - 1, c_names, warp_mask_dirs)

        if opt.tensorboard_dir and (step + 1) % opt.display_count == 0:
            visuals = [
                [im_h, silhouette, im_cocopose] + maybe_densepose,
                [c, warped_cloth, im_c],
                [warped_grid, (warped_cloth + im) * 0.5, im],
            ]
            board_add_images(board, "combine", visuals, step + 1)


def test_tom(opt, test_loader, model, board):
    device = torch.device("cuda", opt.gpu_ids[0])
    model.to(device)
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    save_root = os.path.join(opt.result_dir, base_name, opt.datamode)
    print("Dataset size: %05d!" % (len(test_loader.dataset)), flush=True)

    pbar = tqdm(enumerate(test_loader))
    for step, inputs in pbar:
        dataset_names = inputs["dataset_name"]
        # use subfolders for each subdataset
        try_on_dirs = [osp.join(save_root, dname, "try-on") for dname in dataset_names]

        im_names = inputs["im_name"]
        # if we already did a forward-pass on this batch, skip it
        save_paths = get_save_paths(im_names, try_on_dirs)
        if all(os.path.exists(s) for s in save_paths):
            tqdm.write(f"Skipping {save_paths}")
            continue

        pbar.set_description(im_names[0])

        im = inputs["image"].to(device)
        im_cocopose = inputs["im_cocopose"]
        im_h = inputs["im_head"]
        silhouette = inputs["silhouette"]
        maybe_densepose = (
            [inputs["densepose"].to(device)] if "densepose" in inputs else []
        )

        agnostic = inputs["agnostic"].to(device)
        c = inputs["cloth"].to(device)
        cm = inputs["cloth_mask"].to(device)

        p_rendered, m_composite, p_tryon = model(agnostic, c)

        visuals = [
            [im_h, silhouette, im_cocopose] + maybe_densepose,
            [c, 2 * cm - 1, m_composite],
            [p_rendered, p_tryon, im],
        ]

        save_images(p_tryon, im_names, try_on_dirs)

        if opt.tensorboard_dir and (step + 1) % opt.display_count == 0:
            board_add_images(board, "combine", visuals, step + 1)


def main():
    options_object = TestOptions()
    opt = options_object.parse()
    logger.setLevel(getattr(logging, opt.loglevel.upper()))
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    train_dataset = find_dataset_using_name(opt.dataset)(opt)

    # create dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=False
    )

    # visualization
    board = None
    # Disable Tensorboard for test
    # if opt.tensorboard_dir:
    #     os.makedirs(opt.tensorboard_dir, exist_ok=True)
    #     board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
    #     board.add_text("options", options_object.options_formatted_str)

    # create model & train
    if opt.stage == "GMM":
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, train_loader, model, board)
    elif opt.stage == "TOM":
        model = TOM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, train_loader, model, board)
    else:
        raise NotImplementedError("Model [%s] is not implemented" % opt.stage)

    print("Finished test %s, named: %s!" % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
