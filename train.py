# coding=utf-8
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint
from datasets import (
    get_dataset_class,
    CPDataLoader,
)

from tensorboardX import SummaryWriter
from visualization import board_add_images


def get_opt():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="0", help="comma separated of which GPUs to train on")
    parser.add_argument("-j", "--workers", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=4)

    parser.add_argument("--viton_dataroot", default="data")
    parser.add_argument("--vvt_dataroot", default="/data_hdd/fw_gan_vvt")
    parser.add_argument("--mpv_dataroot", default="/data_hdd/mpv_competition")
    parser.add_argument("--datamode", default="train")
    parser.add_argument(
        "--dataset", choices=("viton", "viton_vvt_mpv", "vvt", "mpv"), default="cp"
    )
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
        help="save tensorboard infos",
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
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=20)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action="store_true", help="shuffle input data")

    opt = parser.parse_args()
    opt.gpu_ids = [int(id) for id in opt.gpu_ids.split(",")]
    return opt


def train_gmm(opt, train_loader, model, board):
    device = torch.device("cuda:0")
    model.to(device)
    #model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0
        - max(0, step - opt.keep_step) / float(opt.decay_step + 1),
    )

    pbar = tqdm(range(opt.keep_step + opt.decay_step), unit="step")
    for step in pbar:
        inputs = train_loader.next_batch()

        im = inputs["image"].to(device) #.cuda()
        im_cocopose = inputs["im_cocopose"].to(device) #.cuda()
        im_h = inputs["im_head"].to(device) #.cuda()
        silhouette = inputs["silhouette"].to(device) #.cuda()
        agnostic = inputs["agnostic"].to(device) #.cuda()
        c = inputs["cloth"].to(device) #.cuda()
        cm = inputs["cloth_mask"].to(device) #.cuda()
        im_c = inputs["im_cloth"].to(device) #.cuda()
        im_g = inputs["grid_vis"].to(device) #.cuda()
        
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode="border")
        # warped_mask = F.grid_sample(cm, grid, padding_mode="zeros")
        warped_grid = F.grid_sample(im_g, grid, padding_mode="zeros")

        visuals = [
            [im_h, silhouette, im_cocopose],
            [c, warped_cloth, im_c],
            [warped_grid, (warped_cloth + im) * 0.5, im],
        ]

        loss = criterionL1(warped_cloth, im_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss: {loss.item():4f}")
        if board and (step + 1) % opt.display_count == 0:
            board_add_images(board, "combine", visuals, step + 1)
            board.add_scalar("metric", loss.item(), step + 1)
            tqdm.write(f'step: {step + 1:8d}, loss: {loss.item():4f}')


        if (step + 1) % opt.save_count == 0:
            save_checkpoint(
                model,
                os.path.join(
                    opt.checkpoint_dir, opt.name, "step_%06d.pth" % (step + 1)
                ),
            )


def train_tom(opt, train_loader, model, board):
    device = torch.device("cuda:0")
    model.to(device)
    #model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0
        - max(0, step - opt.keep_step) / float(opt.decay_step + 1),
    )

    pbar = tqdm(range(opt.keep_step + opt.decay_step))
    for step in pbar:
        inputs = train_loader.next_batch()

        im = inputs["image"].to(device) #.cuda()
        im_cocopose = inputs["im_cocopose"]
        densepose = inputs["densepose"]
        im_h = inputs["im_head"]
        silhouette = inputs["silhouette"]

        agnostic = inputs["agnostic"].to(device)# .cuda()
        c = inputs["cloth"].to(device) #.cuda()
        cm = inputs["cloth_mask"].to(device) #.cuda()
        
        concat_tensor = torch.cat([agnostic, c], 1)
        concat_tensor = concat_tensor.to(device) 

        outputs = model(concat_tensor)
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

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

        tqdm.set_description(
            f"loss: {loss.item():.4f}, l1: {loss_l1.item():.4f}, vgg: {loss_vgg.item():.4f}, mask: {loss_mask.item():.4f}",
        )
        if board and (step + 1) % opt.display_count == 0:
            board_add_images(board, "combine", visuals, step + 1)
            board.add_scalar("metric", loss.item(), step + 1)
            board.add_scalar("L1", loss_l1.item(), step + 1)
            board.add_scalar("VGG", loss_vgg.item(), step + 1)
            board.add_scalar("MaskL1", loss_mask.item(), step + 1)
            print(
                f"step: {step + 1:8d}, loss: {loss.item():.4f}, l1: {loss_l1.item():.4f}, vgg: {loss_vgg.item():.4f}, mask: {loss_mask.item():.4f}",
                flush=True,
            )

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(
                model,
                os.path.join(
                    opt.checkpoint_dir, opt.name, "step_%06d.pth" % (step + 1)
                ),
            )


def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    train_dataset = get_dataset_class(opt.dataset)(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    board = None
    if opt.tensorboard_dir and not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))


    # create model & train & save the final checkpoint
    if opt.stage == "GMM":
        model = GMM(opt)
        model.opt = opt
        if not opt.checkpoint == "" and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(
            model, os.path.join(opt.checkpoint_dir, opt.name, "gmm_final.pth")
        )
    elif opt.stage == "TOM":
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        model.opt = opt
        if not opt.checkpoint == "" and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        train_tom(opt, train_loader, model, board)
        save_checkpoint(
            model, os.path.join(opt.checkpoint_dir, opt.name, "tom_final.pth")
        )
    else:
        raise NotImplementedError("Model [%s] is not implemented" % opt.stage)

    print("Finished training %s, nameed: %s!" % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
