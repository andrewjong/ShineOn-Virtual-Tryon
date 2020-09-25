import os.path as osp
from glob import glob

from datasets.vvt_dataset import VVTDataset


# Testing only
class VVTListDataset(VVTDataset):
    def __init__(self, opt):
        self.data_list = opt.data_list
        self.image_paths = []
        self.cloth_paths = []
        super().__init__(opt)

    def load_file_paths(self):
        self.root = self.opt.vvt_dataroot  # override this
        # make list of
        # cloth <---> image
        with open(self.data_list, "r") as f:
            for line in f:
                # image dir should be our GFLA result
                # we need to Dress cloth_id to image_dir
                image_dir, cloth_id, pose_dir = line.strip().split()
                image_paths = sorted(
                    glob(f"{self.root}/lip_test_frames/{image_dir}/*.png")
                )

                if self.opt.stage == "GMM":
                    # copies the same source cloth_file for the number of test frames
                    cloth_file = glob(
                        f"{self.root}/lip_clothes_person/{cloth_id}/*cloth*"
                    )[0]
                    cloth_paths = [cloth_file] * len(image_paths)
                elif self.opt.stage == "TOM":
                    # take cloth files warped to the test frames; for combination with the test frames.
                    # here we switch from cloth_id to image_dir, because we saved warped clothes to the person id
                    cloth_paths = sorted(
                        glob(f"{self.root}/warp-cloth/{image_dir}/*.png")
                    )

                assert len(image_paths) == len(
                    cloth_paths
                ), f"lens don't match on {image_dir}"
                self.image_paths.extend(image_paths)
                self.cloth_paths.extend(cloth_paths)

    def __len__(self):
        return len(self.image_paths)

    def get_person_image_path(self, index):
        return self.image_paths[index]

    def get_input_cloth_path(self, index):
        return self.cloth_paths[index]

    def get_input_cloth_name(self, index):
        # in test stage, use the folder id of the person. because the clothes will match the person
        image_path = self.get_person_image_path(index)
        folder_id = VVTDataset.extract_video_id(image_path)
        cloth_path = self.get_input_cloth_path(index)
        base_cloth_name = osp.basename(cloth_path)
        frame_name = osp.basename(self.get_person_image_name(index))
        # e.g. 4he21d00f-g11/4he21d00f-g11@10=cloth_front.jpg
        name = osp.join(folder_id, f"{base_cloth_name}.FOR.{frame_name}")
        return name
