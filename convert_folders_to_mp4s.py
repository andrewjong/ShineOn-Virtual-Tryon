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
    parser.add_argument(
        "--same_out_dir",
        action="store_true",
        help="output videos in the same root directory that the video folders are in",
    )
    parser.add_argument("--force", action="store_true", help="Don't warn for overwrite")
    parser.add_argument("--fps", type=float, default=30.0)

    args = parser.parse_args()

    out_dir = args.dataroot if args.same_out_dir else args.out_dir

    try:
        os.makedirs(out_dir, exist_ok=False)
    except FileExistsError:
        if not args.force:
            print(
                f"{out_dir} already exists!"
            )
            choice = input("Are you sure you want to continue? (y/N): ")
            if choice.lower().strip() != "y":
                exit(1)
    print("Writing mp4s under", out_dir)

    video_folders = sorted(next(os.walk(args.dataroot))[1])
    pbar = tqdm(video_folders)
    for video_folder in pbar:

        frames = sorted(os.listdir(os.path.join(args.dataroot, video_folder)))
        if frames:
            frame_path = os.path.join(args.dataroot, video_folder, frames[0])
            height, width, _ = cv2.imread(frame_path).shape

            out_path = os.path.join(out_dir, f"{video_folder}.mp4")
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
