from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error
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
        "-g2", "--generated_dir_path_2", type=str,
        default="/home/gk32721/2021-wacv-video-vton/result/VVTDataset/try-on"
    )
    parser.add_argument(
        "--experiment_one", type=str,
        default="3.1"
    )
    parser.add_argument(
        "--experiment_two", type=str,
        default="3.2"
    )
    parser.add_argument(
        "--result", type=str, default=""
    )
    return parser.parse_args()

def plot(args, df, video_folder):
    df_frames = df.columns[1:]
    indices = range(len(df_frames))
    ssim_generated_1, ssim_generated_2 = df.loc["ssim_truth_generated_1"], df.loc["ssim_truth_generated_2"]
    psnr_generated_1, psnr_generated_2 = df.loc["psnr_truth_generated_1"], df.loc["psnr_truth_generated_2"]
    ssim_generated_1, ssim_generated_2 = ssim_generated_1.tolist()[1:], ssim_generated_2.tolist()[1:]
    psnr_generated_1, psnr_generated_2 = psnr_generated_1.tolist()[1:], psnr_generated_2.tolist()[1:]

    ssim_generated_1, ssim_generated_2 = [float(x) for x in ssim_generated_1], [float(x) for x in ssim_generated_2]
    psnr_generated_1, psnr_generated_2 = [float(x) for x in psnr_generated_1], [float(x) for x in psnr_generated_2]

    figs, axs = plt.subplots(2)
    plt.suptitle(df.loc["file_path"][1].split('/')[0] + " Plot")
    axs[0].plot(indices, ssim_generated_1, label="ssim_truth_generated_1", color="b")
    axs[0].plot(indices, ssim_generated_2, label="ssim_truth_generated_2", color="g")
    axs[1].plot(indices, psnr_generated_1, label="psnr_truth_generated_1", color="r")
    axs[1].plot(indices, psnr_generated_2, label="psnr_truth_generated_2", color="y")
    axs[0].legend()
    axs[1].legend()
    plt.xlabel("# of Frames")
    plt.savefig(f"{args.result}/{args.experiment_one}_{args.experiment_two}_{video_folder}_plot.png")
    plt.close()

def main():
    args = argparser()
    num_errors = 0
    generated_image_frame_paths = glob(osp.join(args.generated_dir_path_1, '**', '*.png'))
    metrics = {} # key: folder/file; value: (ssim, psnr)
    assert len(generated_image_frame_paths) > 0
    print(len(generated_image_frame_paths))
    #from IPython import embed; embed()
    #for root, folder, file in os.walk(args.generated_dir_path):
    generated_video_folder_list_1 = sorted(os.listdir(args.generated_dir_path_1))
    if not osp.exists(args.result):
        os.mkdir(args.result)
    for video_folder in generated_video_folder_list_1:
        out = cv2.VideoWriter(f'{args.result}/compare_{args.experiment_one}_{args.experiment_two}_{video_folder}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20.0, (768, 256))
        generated_image_list = sorted(os.listdir(osp.join(args.generated_dir_path_1, video_folder)))
        for i, image_file in enumerate(generated_image_list):
            ground_truth_image_path = osp.join(args.ground_truth_dir_path, video_folder, image_file)
            generated_image_frame_path_1 = osp.join(args.generated_dir_path_1, video_folder, image_file)
            generated_image_frame_path_2 = osp.join(args.generated_dir_path_2, video_folder, image_file)
            print(generated_image_frame_path_1)
            print(generated_image_frame_path_2)
            print(ground_truth_image_path)
            assert osp.exists(generated_image_frame_path_1)
            assert osp.exists(generated_image_frame_path_2)
            assert osp.exists(ground_truth_image_path)
            ground_truth_image = cv2.imread(ground_truth_image_path)
            generated_image_1 = cv2.imread(generated_image_frame_path_1)
            generated_image_2 = cv2.imread(generated_image_frame_path_2)

            list_of_generated_images = [generated_image_1, generated_image_2]
            details_numpy = np.zeros_like(generated_image_1)
            #calculate metrics
            print(type(generated_image_1))
            print(type(ground_truth_image))

            if generated_image_1 is None or generated_image_2 is None:
                num_errors += 1
            else:
                ssim_metric = [ssim(ground_truth_image, generated_image,
                                   data_range=generated_image.max() - generated_image.min(),
                                   multichannel=True) for generated_image in list_of_generated_images]
                psnr_metric = [psnr(ground_truth_image, generated_image,
                                   data_range=generated_image.max() - generated_image.min())
                               for generated_image in list_of_generated_images]

                # combine frames
                combined_image = np.concatenate((ground_truth_image, list_of_generated_images[0], list_of_generated_images[1], details_numpy), axis=1)
                #from IPython import embed;
                #embed()

                print(combined_image.shape)

                cv2.putText(combined_image, f"Ground Truth Image", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0))
                cv2.putText(combined_image, f"Generated Image 1", (210, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0))
                cv2.putText(combined_image, f"Generated Image 2", (450, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0))
                cv2.putText(combined_image, f"SSIM_1 Metric: {ssim_metric[0]}", (600, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255))
                cv2.putText(combined_image, f"SSIM_2 Metric: {ssim_metric[1]}", (600, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255))
                cv2.putText(combined_image, f"PSNR_1 Metric: {psnr_metric[0]}", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255))
                cv2.putText(combined_image, f"PSNR_2 Metric: {psnr_metric[1]}", (600, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255))
                cv2.putText(combined_image, f"Frame: {i}", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255))
                #cv2.imwrite('combined_image.jpg', combined_image)
                cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

                # add frames to video
                out.write(combined_image)
                # save in dictionary
                metrics[i] = (osp.join(video_folder, image_file), ssim_metric[0], ssim_metric[1], psnr_metric[0], psnr_metric[1])
                print("calculated")
                print(ssim_metric)
                print(psnr_metric)

        out.release()
        df = pd.DataFrame.from_dict(metrics)
        df.index = ["file_path", "ssim_truth_generated_1", "ssim_truth_generated_2", "psnr_truth_generated_1", "psnr_truth_generated_2"]
        plot(args, df, video_folder)
        #print(f"{args.result}/compare_{args.experiment_one}_{args.experiment_two}_{video_folder}/{args.experiment_one}_{args.experiment_two}_metrics_output.csv")
        #assert 1 == 0, os.getcwd()
        df.to_csv(f"{args.result}/{args.experiment_one}_{args.experiment_two}_{video_folder}_metrics_output.csv")

if __name__ == '__main__':
    main()
