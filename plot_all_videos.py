from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
from glob import glob
import os.path as osp
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--ground_truth_dir_path", "--truth", type=str,
        default="/home/gk32721/data/fw_gan_vvt/train/train_frames"
    )
    parser.add_argument(
        "-g1", "--generated_dir_path_1", type=str,
        default="/home/gk32721/2021-wacv-video-vton/result/VVTDataset/try-on"
    )

    parser.add_argument(
        "--experiment_one", type=str,
        default="3.1"
    )

    parser.add_argument(
        "--result", type=str, default=""
    )
    return parser.parse_args()

def plot(args, df, video_folder):
    from IPython import embed; embed()
    df_frames = df.columns[1:]
    indices = range(len(df_frames))
    ssim_generated_1 = df.loc["ssim_truth_generated_1"]
    psnr_generated_1 = df.loc["psnr_truth_generated_1"]
    ssim_generated_1 = ssim_generated_1.tolist()[1:]
    psnr_generated_1 = psnr_generated_1.tolist()[1:]

    ssim_generated_1 = [float(x) for x in ssim_generated_1]
    psnr_generated_1 = [float(x) for x in psnr_generated_1]

    figs, axs = plt.subplots(2)
    plt.suptitle(df.loc["file_path"][1].split('/')[0] + " Plot")
    axs[0].plot(indices, ssim_generated_1, label="ssim_truth_generated_1", color="b")
    axs[1].plot(indices, psnr_generated_1, label="psnr_truth_generated_1", color="r")
    axs[0].legend()
    axs[1].legend()
    plt.xlabel("# of Frames")
    plt.savefig(f"{args.result}/{args.experiment_one}_{video_folder}_plot.png")
    plt.close()

def main():
    args = argparser()
    num_errors = 0
    generated_image_frame_paths = glob(osp.join(args.generated_dir_path_1, '**', '*.png'))
    df = pd.DataFrame(index=[], columns=[])
    assert len(generated_image_frame_paths) > 0
    print(len(generated_image_frame_paths))
    generated_video_folder_list_1 = sorted(os.listdir(args.generated_dir_path_1))
    if not osp.exists(args.result):
        os.mkdir(args.result)
    figs, axs = plt.subplots(2)
    max = 0
    step = 0
    for video_folder in generated_video_folder_list_1:
        generated_image_list = sorted(os.listdir(osp.join(args.generated_dir_path_1, video_folder)))
        metrics = []
        #psnr_metrics = []
        for i, image_file in enumerate(generated_image_list):
            ground_truth_image_path = osp.join(args.ground_truth_dir_path, video_folder, image_file)
            generated_image_frame_path_1 = osp.join(args.generated_dir_path_1, video_folder, image_file)
            print(generated_image_frame_path_1)
            print(ground_truth_image_path)
            assert osp.exists(generated_image_frame_path_1)
            assert osp.exists(ground_truth_image_path)
            ground_truth_image = cv2.imread(ground_truth_image_path)
            generated_image_1 = cv2.imread(generated_image_frame_path_1)

            #calculate metrics
            print(type(generated_image_1))
            print(type(ground_truth_image))

            if generated_image_1 is None:
                num_errors += 1
            else:
                ssim_metric = ssim(ground_truth_image, generated_image_1,
                                   data_range=generated_image_1.max() - generated_image_1.min(),
                                   multichannel=True)
                psnr_metric = psnr(ground_truth_image, generated_image_1,
                                   data_range=generated_image_1.max() - generated_image_1.min())


                # save in dictionary
                #metrics[video_folder] = (osp.join(image_file), ssim_metric, psnr_metric)
                step += 1
                metrics.append((ssim_metric, psnr_metric))
                print("calculated")
                print(ssim_metric)
                print(psnr_metric)
        latest_series = pd.Series(metrics, index=list(range(len(metrics))), name=video_folder)
        df = pd.concat([df,latest_series], ignore_index=True, axis=1)
        if len(metrics) > max:
            max = len(metrics)
        print("max", max)
        #plt.plot()
        #axs[0].plot(list(range(len(generated_image_list))), ssim_metrics, label="ssim_truth_generated_1", color="b")
        #axs[1].plot(list(range(len(generated_image_list))), psnr_metrics, label="psnr_truth_generated_1", color="r")
    #axs[0].legend()
    #axs[1].legend()
    #plt.xlabel("# of Frames")
    #plt.savefig(f"{args.result}/{args.experiment_one}_plot.png")
    #plt.show()
    #plt.close()
    print("num_errors", num_errors)
    #df = pd.DataFrame.from_dict(metrics)
    df.columns = generated_video_folder_list_1
    #df.index = ["file_path", "ssim_truth_generated_1", "psnr_truth_generated_1"]
    #plot(args, df, video_folder)
    df.to_csv(f"{args.result}/{args.experiment_one}_metrics_output.csv")

if __name__ == '__main__':
    main()