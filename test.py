#coding=utf-8
import argparse
import os
import os.path as osp
from os.path import basename
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm

from datasets import (
    get_dataset_class,
    CPDataLoader,
    DATASETS)
from networks import GMM, UnetGenerator, load_checkpoint
from visualization import board_add_images, save_images, get_save_paths


def get_opt():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--vvt_dataroot", default="/data_hdd/vvt_competition")
    parser.add_argument("--mpv_dataroot", default="/data_hdd/mpv_competition")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument( "--dataset", choices=DATASETS.keys(), default="cp" )
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--tensorboard_dir', type=str, help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default = 1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)

    save_root = os.path.join(opt.result_dir, base_name, opt.datamode)

    pbar = tqdm(enumerate(test_loader.data_loader), total=len(test_loader.data_loader))
    for step, inputs in pbar:
        dataset_names = inputs["dataset_name"]
        # produce subfolders for each subdataset
        warp_cloth_dirs = [osp.join(save_root, dname, "warp-cloth") for dname in dataset_names]
        warp_mask_dirs = [osp.join(save_root, dname, "warp-mask") for dname in dataset_names]

        c_names = inputs["c_name"]
        # if we already did a forward-pass on this batch, skip it
        save_paths = get_save_paths(c_names, warp_cloth_dirs)
        if all(os.path.exists(s) for s in save_paths):
            pbar.set_description(f"Skipping {c_names[0]}")
            continue

        pbar.set_description(c_names[0])
        # unpack the rest of the data
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()

        # forward pass
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        # save images
        save_images(warped_cloth, c_names, warp_cloth_dirs)
        save_images(warped_mask*2-1, c_names, warp_mask_dirs)

        if opt.tensorboard_dir and (step+1) % opt.display_count == 0:
            visuals = [[im_h, shape, im_pose],
                       [c, warped_cloth, im_c],
                       [warped_grid, (warped_cloth + im) * 0.5, im]]
            board_add_images(board, 'combine', visuals, step+1)



def test_tom(opt, test_loader, model, board):
    model.cuda()
    model.eval()
    
    base_name = os.path.basename(opt.checkpoint)
    save_root = os.path.join(opt.result_dir, base_name, opt.datamode)
    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)

    pbar = tqdm(enumerate(test_loader.data_loader), total=len(test_loader.data_loader))
    for step, inputs in pbar:
        dataset_names = inputs["dataset_name"]
        # use subfolders for each subdataset
        try_on_dirs = [osp.join(save_root, dname, "try-on") for dname in dataset_names]

        im_names = inputs['im_name']
        # if we already did a forward-pass on this batch, skip it
        save_paths = get_save_paths(im_names, try_on_dirs)
        if all(os.path.exists(s) for s in save_paths):
            tqdm.write(f"Skipping {save_paths}")
            continue

        pbar.set_description(im_names[0])

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        
        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, 2*cm-1, m_composite], 
                   [p_rendered, p_tryon, im]]
            
        save_images(p_tryon, im_names, try_on_dirs)

        if opt.tensorboard_dir and (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)


def main():
    opt = get_opt()
    print(opt)
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    train_dataset = get_dataset_class(opt.dataset)(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    board = None
    if opt.tensorboard_dir:
        if not os.path.exists(opt.tensorboard_dir):
            os.makedirs(opt.tensorboard_dir)
        board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train
    if opt.stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, train_loader, model, board)
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, train_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
  
    print('Finished test %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
