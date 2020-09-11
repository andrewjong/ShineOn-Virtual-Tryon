from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error
import argparse
from glob import glob
import os.path as osp
import cv2
import pandas as pd

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth_dir_path", type=str,
        default="/home/gk32721/data/fw_gan_vvt/train/train_frames"
    )
    parser.add_argument(
        "--generated_dir_path", type=str,
        default="/home/gk32721/2021-wacv-video-vton/result/VVTDataset/try-on"
    )
    return parser.parse_args()

def main():
    args = argparser()
    generated_image_frame_paths = glob(osp.join(args.generated_dir_path, '**', '*.png'))
    metrics = {} # key: folder/file; value: (ssim, psnr)
    assert len(generated_image_frame_paths) > 0
    print(len(generated_image_frame_paths))
    for generated_image_frame_path in generated_image_frame_paths:
        folder, file = generated_image_frame_path.split('/')[-2:]
        ground_truth_image_path = osp.join(args.ground_truth_dir_path, folder, file)

        generated_image = cv2.imread(generated_image_frame_path)
        ground_truth_image = cv2.imread(ground_truth_image_path)
        print(type(generated_image), generated_image.shape)
        print(type(ground_truth_image), ground_truth_image.shape)
        ssim_metric = ssim(ground_truth_image, generated_image,
                                  data_range=generated_image.max() - generated_image.min(),
                                  multichannel=True)
        psnr_metric = psnr(ground_truth_image, generated_image,
                           data_range=generated_image.max() - generated_image.min())
        # save in dictionary
        metrics[osp.join(folder, file)] = (ssim_metric, psnr_metric)
        print("calculated")
        print(ssim_metric)
        print(psnr_metric)
    df = pd.DataFrame.from_dict(metrics)
    df.to_csv("vanilla_unet_cocopose_metrics_output.csv")
        #generated_image_numpy =
        #ground_truth_image_numpy


if __name__ == '__main__':
    main()
