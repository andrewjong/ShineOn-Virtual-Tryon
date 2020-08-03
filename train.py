# coding=utf-8
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import log
from datasets import find_dataset_using_name
from datasets.n_frames_interface import maybe_combine_frames_and_channels
from networks.cpvton import (
    GMM,
    VGGLoss,
    load_checkpoint,
    save_checkpoint,
    TOM,
)
from options.train_options import TrainOptions
from visualization import board_add_images

logger = log.setup_custom_logger("logger")


def train_gmm(opt, train_loader, model, board):
    device = torch.device("cuda", opt.gpu_ids[0])
    model.to(device)
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: 1.0
        - max(0, e - opt.keep_epochs) / float(opt.decay_epochs + 1),
    )

    steps = 0
    for epoch in tqdm(
        range(opt.keep_epochs + opt.decay_epochs), desc="Epoch", unit="epoch"
    ):

        pbar = tqdm(enumerate(train_loader), unit="step")
        for i, inputs in pbar:

            # ensure epoch is over when steps is divisible by datacap
            if i >= opt.datacap:
                logger.info(f"Reached dataset cap {opt.datacap}")
                break
            inputs = maybe_combine_frames_and_channels(opt, inputs)
            im = inputs["image"].to(device)
            im_cocopose = inputs["im_cocopose"].to(device)
            maybe_densepose = (
                [inputs["densepose"].to(device)] if "densepose" in inputs else []
            )
            im_h = inputs["im_head"].to(device)
            silhouette = inputs["silhouette"].to(device)
            agnostic = inputs["agnostic"].to(device)
            c = inputs["cloth"].to(device)
            im_c = inputs["im_cloth"].to(device)
            im_g = inputs["grid_vis"].to(device)

            grid, theta = model(agnostic, c)
            warped_cloth = F.grid_sample(c, grid, padding_mode="border")
            # warped_mask = F.grid_sample(cm, grid, padding_mode="zeros")
            warped_grid = F.grid_sample(im_g, grid, padding_mode="zeros")

            visuals = [
                [im_h, silhouette, im_cocopose] + maybe_densepose,
                [c, warped_cloth, im_c],
                [warped_grid, (warped_cloth + im) * 0.5, im],
            ]

            loss = criterionL1(warped_cloth, im_c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"loss: {loss.item():4f}")
            if board and steps % opt.display_count == 0:
                board_add_images(board, "combine", visuals, steps)
                board.add_scalar("epoch", epoch, steps)
                board.add_scalar("metric", loss.item(), steps)
                logger.info(f"step: {steps:8d}, loss: {loss.item():4f}")
            steps += 1

        if epoch % opt.save_count == 0:
            save_checkpoint(
                model,
                os.path.join(opt.checkpoint_dir, opt.name, f"epoch_{epoch:04d}.pth"),
            )
        scheduler.step()


def train_tom(opt, train_loader, model, board):
    torch.cuda.set_device(opt.gpu_ids[0])
    device = torch.device("cuda", opt.gpu_ids[0])
    model.to(device)
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: 1.0
        - max(0, e - opt.keep_epochs) / float(opt.decay_epochs + 1),
    )

    steps = 0
    for epoch in tqdm(
        range(opt.keep_epochs + opt.decay_epochs), desc="Epoch", unit="epoch"
    ):
        pbar = tqdm(enumerate(train_loader), unit="step")
        for i, inputs in pbar:
            if i >= opt.datacap:
                logger.info(f"Reached dataset cap {opt.datacap}")
                break
            inputs = maybe_combine_frames_and_channels(opt, inputs)
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

            p_rendered, m_composite, p_tryon = model(agnostic, c)

            visuals = [
                [im_h, silhouette, im_cocopose] + maybe_densepose,
                [c, cm * 2 - 1, m_composite * 2 - 1],
                [p_rendered, p_tryon, im],
            ]

            loss_l1 = criterionL1(p_tryon, im)
            loss_vgg = criterionVGG(p_tryon, im)
            loss_mask = criterionMask(m_composite, cm)
            loss = loss_l1 + loss_vgg + loss_mask
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                desc=f"loss: {loss.item():.4f}, l1: {loss_l1.item():.4f}, vgg: {loss_vgg.item():.4f}, mask: {loss_mask.item():.4f}",
            )
            if board and steps % opt.display_count == 0:
                board_add_images(board, "combine", visuals, steps)
                board.add_scalar("epoch", epoch, steps)
                board.add_scalar("metric", loss.item(), steps)
                board.add_scalar("L1", loss_l1.item(), steps)
                board.add_scalar("VGG", loss_vgg.item(), steps)
                board.add_scalar("MaskL1", loss_mask.item(), steps)
                logger.info(
                    f"step: {steps:8d}, loss: {loss.item():.4f}, l1: {loss_l1.item():.4f}, vgg: {loss_vgg.item():.4f}, mask: {loss_mask.item():.4f}",
                    flush=True,
                )
            steps += 1

        if epoch % opt.save_count == 0:
            save_checkpoint(
                model,
                os.path.join(opt.checkpoint_dir, opt.name, f"epoch_{epoch:04d}.pth"),
            )
        scheduler.step()


def main():
    options_object = TrainOptions()
    opt = options_object.parse()
    logger.setLevel(getattr(logging, opt.loglevel.upper()))
    logger.info(f"Start to train stage: {opt.stage}, named: {opt.name}!")

    # create dataset
    train_dataset = find_dataset_using_name(opt.dataset)(opt)

    # create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        shuffle=not opt.no_shuffle,
    )

    # visualization
    board = None
    if opt.tensorboard_dir:
        os.makedirs(opt.tensorboard_dir, exist_ok=True)
        board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
        board.add_text("options", options_object.options_formatted_str)

    # create model & train & save the final checkpoint
    if opt.stage == "GMM":
        model = GMM(opt)
        train_fn = train_gmm
        final_save = "gmm_final.pth"
    elif opt.stage == "TOM":
        model = TOM(opt)
        train_fn = train_tom
        final_save = "tom_final.pth"
    else:
        raise NotImplementedError(f"Model [{opt.stage}] is not implemented")

    if opt.checkpoint and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
    if torch.cuda.device_count() > 1 and opt.dataparallel:
        model = nn.DataParallel(model)
    train_fn(opt, train_loader, model, board)
    save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, final_save))

    logger.info(f"Finished training {opt.stage}, named: {opt.name}!")


if __name__ == "__main__":
    main()
