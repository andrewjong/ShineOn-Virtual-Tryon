# coding=utf-8
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
from networks.cpvton import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint, TOM
from datasets import get_dataset_class

from tensorboardX import SummaryWriter
from visualization import board_add_images


def get_opt():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", default="GMM")
    parser.add_argument(
        "--gpu_ids", default="0", help="comma separated of which GPUs to train on"
    )
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("-b", "--batch_size", type=int, default=8)

    parser.add_argument("--viton_dataroot", default="data")
    parser.add_argument("--vvt_dataroot", default="/data_hdd/fw_gan_vvt")
    parser.add_argument("--mpv_dataroot", default="/data_hdd/mpv_competition")
    parser.add_argument("--datamode", default="train")
    parser.add_argument(
        "--dataset", choices=("viton", "viton_vvt_mpv", "vvt", "mpv"), default="cp"
    )
    parser.add_argument("--data_parallel", type=int, default=0)
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="initial learning rate for adam"
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="tensorboard",
        help="save tensorboard infos. pass empty string '' to disable tensorboard",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="save checkpoint infos",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="model checkpoint for initialization"
    )
    parser.add_argument(
        "--display_count",
        type=int,
        help="how often to update tensorboard, in steps",
        default=100,
    )
    parser.add_argument(
        "--save_count",
        type=int,
        help="how often to save a checkpoint, in epochs",
        default=1,
    )
    parser.add_argument(
        "--keep_epochs",
        type=int,
        help="number of epochs with initial learning rate",
        default=100,
    )
    parser.add_argument(
        "--decay_epochs",
        type=int,
        help="number of epochs to linearly decay the learning rate",
        default=100,
    )
    parser.add_argument(
        "--datacap",
        type=float,
        default=float("inf"),
        help="limits the dataset to this many batches",
    )
    parser.add_argument("--shuffle", action="store_true", help="shuffle input data")

    opt = parser.parse_args()
    opt.gpu_ids = [int(id) for id in opt.gpu_ids.split(",")]
    return opt


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
        pbar = tqdm(train_loader, unit="step")
        for inputs in pbar:
            if steps > opt.datacap:
                tqdm.write(f"Reached dataset cap {opt.datacap}")
                break
            im = inputs["image"].to(device)
            im_cocopose = inputs["im_cocopose"].to(device)
            densepose = inputs["densepose"]
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
                [im_h, silhouette, im_cocopose, densepose],
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
                tqdm.write(f"step: {steps:8d}, loss: {loss.item():4f}")
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
        pbar = tqdm(train_loader, unit="step")
        for inputs in pbar:
            if steps > opt.datacap:
                tqdm.write(f"Reached dataset cap {opt.datacap}")
                break
            im = inputs["image"].to(device)
            im_cocopose = inputs["im_cocopose"].to(device)
            densepose = inputs["densepose"].to(device)
            im_h = inputs["im_head"].to(device)
            silhouette = inputs["silhouette"].to(device)

            agnostic = inputs["agnostic"].to(device)
            c = inputs["cloth"].to(device)
            cm = inputs["cloth_mask"].to(device)

            p_rendered, m_composite, p_tryon = model(agnostic, c)

            visuals = [
                [im_h, silhouette, im_cocopose, densepose],
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
                print(
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
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    train_dataset = get_dataset_class(opt.dataset)(opt)

    # create dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, num_workers=opt.workers
    )

    # visualization
    board = None
    if opt.tensorboard_dir:
        os.makedirs(opt.tensorboard_dir, exist_ok=True)
        board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

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

    if not opt.checkpoint == "" and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
    if opt.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    train_fn(opt, train_loader, model, board)
    save_checkpoint(
        model, os.path.join(opt.checkpoint_dir, opt.name, final_save)
    )

    print("Finished training %s, named: %s!" % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
