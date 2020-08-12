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
from models import find_model_using_name
from models.networks.cpvton import load_checkpoint, save_checkpoint
from models.base_model import get_and_cat_inputs
from models.networks.loss import VGGLoss


"""    (
    unet_mask_model,
    VGGLoss,
    load_checkpoint,
    save_checkpoint,
    TOM,
)"""
from options.train_options import TrainOptions
from visualization import board_add_images

logger = log.setup_custom_logger("logger")


def train_warp(opt, train_loader, model, board):
    device = torch.device("cuda", opt.gpu_ids[0])
    model.to(device)
    model.train()



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

        pbar = tqdm(enumerate(train_loader), unit="step", total=len(train_loader))
        for i, batch in pbar:

            # ensure epoch is over when steps is divisible by datacap
            if i >= opt.datacap:
                logger.info(f"Reached dataset cap {opt.datacap}")
                break
            batch = maybe_combine_frames_and_channels(opt, batch)
            # unpack
            im_c = batch["im_cloth"].to(device)
            im_g = batch["grid_vis"].to(device)
            im = batch["image"].to(device)
            im_cocopose = batch["im_cocopose"].to(device)
            maybe_densepose = [batch["densepose"].to(device)] if "densepose" in batch else []
            c = batch["cloth"].to(device)
            im_h = batch["im_head"].to(device)
            silhouette = batch["silhouette"].to(device)
            person_inputs = get_and_cat_inputs(batch, opt.person_inputs).to(device)
            cloth_inputs = get_and_cat_inputs(batch, opt.cloth_inputs).to(device)

            # forward
            grid, theta = model(person_inputs, cloth_inputs)
            warped_cloth = F.grid_sample(c, grid, padding_mode="border")
            warped_grid = F.grid_sample(im_g, grid, padding_mode="zeros")
            # loss
            loss = F.l1_loss(warped_cloth, im_c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"loss: {loss.item():4f}")



            ######################################################

            """inputs = maybe_combine_frames_and_channels(opt, inputs)
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
            warped_grid = F.grid_sample(im_g, grid, padding_mode="zeros")"""
            ################################################
            # visualize and logging
            visuals = [
                [im_h, silhouette, im_cocopose] + maybe_densepose,
                [c, warped_cloth, im_c],
                [warped_grid, (warped_cloth + im) * 0.5, im],
            ]


            if board and steps % opt.display_count == 0:
                board_add_images(board, "combine", visuals, steps)
                board.add_scalar("epoch", epoch, steps)
                board.add_scalar("metric", loss.item(), steps)
                logger.info(f"step: {steps:8d}, loss: {loss.item():4f}")
            

            if steps % opt.save_count == 0:
                save_checkpoint(
                    model,
                    os.path.join(opt.checkpoint_dir, opt.name, f"model_epoch_{epoch:04d}_step_{steps:09d}.pth"),
                )
            steps += 1
        scheduler.step()


def train_unet(opt, train_loader, model, board):
    device = torch.device("cuda", opt.gpu_ids[0])
    model.to(device)
    model.train()


    # criterion
    vgg_loss = VGGLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: 1.0
        - max(0, e - opt.keep_epochs) / float(opt.decay_epochs + 1),
    )

    steps = 0
    prev_frame = None
    for epoch in tqdm(
        range(opt.keep_epochs + opt.decay_epochs), desc="Epoch", unit="epoch"
    ):
        pbar = tqdm(enumerate(train_loader), unit="step", total=len(train_loader))
        for i, batch in pbar:
            if i >= opt.datacap:
                logger.info(f"Reached dataset cap {opt.datacap}")
                break

            batch = maybe_combine_frames_and_channels(opt, batch)
            # unpack
            cm = batch["cloth_mask"].to(device)
            flow = batch["flow"].to(device) if opt.flow else None
            im = batch["image"].to(device)
            im_cocopose = batch["im_cocopose"].to(device)
            maybe_densepose = [batch["densepose"].to(device)] if "densepose" in batch else []
            c = batch["cloth"].to(device)
            im_h = batch["im_head"].to(device)
            silhouette = batch["silhouette"].to(device)

            person_inputs = get_and_cat_inputs(batch, opt.person_inputs).to(device)
            cloth_inputs = get_and_cat_inputs(batch, opt.cloth_inputs).to(device)

            # forward
            p_rendered, m_composite, p_tryon = model(person_inputs, cloth_inputs, flow)
            # loss
            loss_l1 = F.l1_loss(p_tryon, im)
            loss_vgg = vgg_loss(p_tryon, im)
            loss_mask = F.l1_loss(m_composite, cm)
            loss = loss_l1 + loss_vgg + loss_mask
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            prev_frame = im
            #return result
            ############################################
            """inputs = maybe_combine_frames_and_channels(opt, inputs)
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

            p_rendered, m_composite, p_tryon = model(agnostic, c)"""
            #########################################################
            visuals = [
                [im_h, silhouette, im_cocopose] + maybe_densepose,
                [c, cm * 2 - 1, m_composite * 2 - 1],
                [p_rendered, p_tryon, im],
            ]

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
                )
            

            if steps % opt.save_count == 0:
                save_checkpoint(
                    model,
                    os.path.join(opt.checkpoint_dir, opt.name, f"model_epoch_{epoch:04d}_step_{steps:09d}.pth"),
                )
            steps += 1
        scheduler.step()


def main(train=True):
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
    # create model
    model_class = find_model_using_name(opt.model)
    if opt.checkpoint or not train:
        model = model_class.load_from_checkpoint(opt.checkpoint)
    else:
        model = model_class(opt)

    # visualization
    board = None
    if opt.tensorboard_dir:
        os.makedirs(opt.tensorboard_dir, exist_ok=True)
        board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
        board.add_text("options", options_object.options_formatted_str)

    # create model & train & save the final checkpoint
    if opt.model == "warp":
        #model = GMM(opt)
        train_fn = train_warp
        final_save = "warp_final.pth"
    elif opt.model == "unet_mask":
        #model = TOM(opt)
        train_fn = train_unet
        final_save = "unet_final.pth"
    else:
        raise NotImplementedError(f"Model [{opt.model}] is not implemented")

    if opt.checkpoint and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
    if torch.cuda.device_count() > 1 and len(opt.gpu_ids) > 1:
        model = nn.DataParallel(model)
    train_fn(opt, train_loader, model, board)
    save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, final_save))

    logger.info(f"Finished training {opt.stage}, named: {opt.name}!")


if __name__ == "__main__":
    main(train=True)
