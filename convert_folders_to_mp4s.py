import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import cv2
from tqdm import tqdm

"""
# tryon_list.txt
cloth_product_path,person_id
"""

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataroot")
    parser.add_argument(
        "--tryon_list",
        help="A CSV file to specify what cloth should be given to each person. "
        "CSV should be formatted as `cloth_product_path, person_id`. ",
    )
    parser.add_argument("--out_dir", default="mp4s")
    parser.add_argument("--fps", type=float, default=30.0)

    args = parser.parse_args()

    try:
        os.makedirs(args.out_dir, exist_ok=False)
    except FileExistsError:
        print(
            f"{args.out_dir} already exists! Not continuing to prevent overwrite. "
            f"Please remove manually."
        )
        exit(1)

    video_folders = sorted(os.listdir(args.dataroot))
    pbar = tqdm(video_folders)
    for video_folder in pbar:

        frames = sorted(os.listdir(os.path.join(args.dataroot, video_folder)))
        if frames:
            frame_path = os.path.join(args.dataroot, video_folder, frames[0])
            height, width, _ = cv2.imread(frame_path).shape

            out_path = os.path.join(args.out_dir, f"{video_folder}.mp4")
            pbar.set_description(out_path)
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height),
            )

            for frame in frames:
                frame_path = os.path.join(args.dataroot, video_folder, frame)
                try:
                    img = cv2.imread(frame_path)
                except:
                    tqdm.write(f"{frame_path} is not a valid image! Skipping!")
                    continue
                writer.write(img)
            writer.release()
        else:
            tqdm.write(f"No frames found for {video_folder: }")

    print("Done!")
