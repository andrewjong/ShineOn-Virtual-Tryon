# coding=utf-8
import logging
import os
import os.path as osp

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import find_dataset_using_name
import log
from networks.cpvton import GMM, load_checkpoint, TOM
from options.test_options import TestOptions
from visualization import board_add_images, save_images, get_save_paths

logger = log.setup_custom_logger("logger")

def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)

    save_root = os.path.join(opt.result_dir, base_name, opt.datamode)

    pbar = tqdm(enumerate(test_loader))
    for step, inputs in pbar:
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
        im = inputs["image"].cuda()
        im_cocopose = inputs["im_cocopose"].cuda()
        densepose = inputs["densepose"].cuda()
        im_h = inputs["im_head"].cuda()
        shape = inputs["silhouette"].cuda()
        agnostic = inputs["agnostic"].cuda()
        c = inputs["cloth"].cuda()
        cm = inputs["cloth_mask"].cuda()
        im_c = inputs["im_cloth"].cuda()
        im_g = inputs["grid_vis"].cuda()

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
                [im_h, shape, im_cocopose, densepose],
                [c, warped_cloth, im_c],
                [warped_grid, (warped_cloth + im) * 0.5, im],
            ]
            board_add_images(board, "combine", visuals, step + 1)


def test_tom(opt, test_loader, model, board):
    model.cuda()
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

        im = inputs["image"].cuda()
        im_cocopose = inputs["im_cocopose"]
        im_h = inputs["im_head"]
        shape = inputs["silhouette"]

        agnostic = inputs["agnostic"].cuda()
        c = inputs["cloth"].cuda()
        cm = inputs["cloth_mask"].cuda()

        p_rendered, m_composite, p_tryon = model(agnostic, c)

        visuals = [
            [im_h, shape, im_cocopose],
            [c, 2 * cm - 1, m_composite],
            [p_rendered, p_tryon, im],
        ]

        save_images(p_tryon, im_names, try_on_dirs)

        if opt.tensorboard_dir and (step + 1) % opt.display_count == 0:
            board_add_images(board, "combine", visuals, step + 1)


def main():
    opt = TestOptions().parse()
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
    if opt.tensorboard_dir:
        if not os.path.exists(opt.tensorboard_dir):
            os.makedirs(opt.tensorboard_dir)
        board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

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
