# coding=utf-8
import argparse
import logging
import os
import os.path as osp
from glob import glob

from datasets.n_frames_interface import NFramesInterface
from datasets.tryon_dataset import TryonDataset

logger = logging.getLogger("logger")


class VVTDataset(TryonDataset, NFramesInterface):
    """ CP-VTON dataset with FW-GAN's VVT folder structure. """

    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser, is_train, shared=False):
        if not shared:
            parser = TryonDataset.modify_commandline_options(parser, is_train)
        parser = NFramesInterface.modify_commandline_options(parser, is_train)
        parser.add_argument("--vvt_dataroot", default="/data_hdd/fw_gan_vvt")
        parser.add_argument(
            "--warp_cloth_dir",
            default="warp-cloth",
            help="path to the GMM-generated intermediary warp-cloth folder for TOM",
        )
        return parser

    @staticmethod
    def extract_folder_id(image_path):
        return image_path.split(os.sep)[-2]

    def __init__(self, opt, i_am_validation=False):
        """

        Args:
            opt: Namespace
        """
        self._video_start_indices = set()
        TryonDataset.__init__(self, opt, i_am_validation)
        NFramesInterface.__init__(self, opt)

    # @overrides(TryonDataset)
    def load_file_paths(self, i_am_validation=False):
        """ Reads the Videos from the fw_gan_vvt dataset. """
        if self.is_train or self.opt.task == "reconstruction":
            self.load_file_paths_for_reconstruction_task(i_am_validation)
        else:
            self.load_file_paths_for_tryon_task()

    def load_file_paths_for_reconstruction_task(self, i_am_validation):
        """ For the reconstruction task (training) We glob for videos """
        self.root = self.opt.vvt_dataroot  # override this
        folder = f"{self.opt.datamode}/{self.opt.datamode}_frames"

        videos_search = f"{self.root}/{folder}/*/"
        video_folders = sorted(glob(videos_search))
        num_videos = len(video_folders)
        validation_index = int((1 - self.val_fraction) * num_videos)

        if i_am_validation:
            start, end = validation_index, num_videos
        else:
            start, end = 0, validation_index

        self.register_videos(video_folders, start, end)

    def register_videos(self, video_folders, start=0, end=-1):
        """ Records what index each video starts at, and collects all the video frames
        in a flat list. """
        for video_folder in video_folders[start:end]:
            self._record_video_start_index()  # starts with 0
            self._add_video_frames_to_image_names(video_folder)

    def load_file_paths_for_tryon_task(self):
        """ For the try-on task, the videos are specified in a csv file """
        self.video_ids_to_cloth_paths = {}
        video_folders = []
        with open(self.opt.tryon_list, "r") as f:
            all_lines = f.readlines()
        for line in all_lines:
            cloth_path, video_id = line.split(",")
            cloth_path, video_id = cloth_path.strip(), video_id.strip()
            self.video_ids_to_cloth_paths[video_id] = cloth_path

            video_folder = osp.join(
                self.opt.vvt_dataroot,
                self.opt.datamode,
                f"{self.opt.datamode}_frames",
                video_id,
            )
            video_folders.append(video_folder)

        self.register_videos(video_folders)

    def _add_video_frames_to_image_names(self, video_folder):
        search = f"{video_folder}/*.png"
        frames = sorted(glob(search))
        self.image_names.extend(frames)

    def _record_video_start_index(self):
        # add the video index
        beg_index = len(self.image_names)
        self._video_start_indices.add(beg_index)

    ########################
    # CLOTH REPRESENTATION
    ########################

    # @overrides(TryonDataset)
    def get_input_cloth_path(self, index):
        image_path = self.image_names[index]
        folder_id = VVTDataset.extract_folder_id(image_path)
        # for some reason fw_gan_vvt's clothes_persons folder is in upper case. this is
        # a temporay hack; we should really lowercase those folders.
        # it also removes the ending sub-id, which is the garment id
        folder_id, cloth_id = folder_id.upper().split("-")

        # TRYON TASK
        if not self.opt.is_train and self.opt.task == "tryon":
            return self.video_ids_to_cloth_paths[folder_id]

        # RECONSTRUCTION TASK
        if self.opt.model == "warp":
            path = osp.join(self.root, "clothes_person", "img")
            keyword = "cloth_front"
        else:
            # TOM
            if self.opt.warp_cloth_dir == "warp-cloth":  # symlink version
                # TODO: this line is specific to our own directory setup. should remove this
                path = osp.join(self.root, self.opt.datamode, "warp-cloth")
            else:  # user specifies the path
                path = self.opt.warp_cloth_dir
            frame_keyword_start = image_path.find("frame_")
            frame_keyword_end = image_path.rfind(".")
            frame_word = image_path[frame_keyword_start:frame_keyword_end]
            keyword = f"cloth_front*{frame_word}"

        cloth_folder = osp.join(path, folder_id)

        search = f"{cloth_folder}/{folder_id}-{cloth_id}*{keyword}.*"
        cloth_path_matches = sorted(glob(search))
        if len(cloth_path_matches) == 0:
            logger.debug(
                f"{search=} not found, relaxing search to any cloth term. We should probably fix this later."
            )
            search = f"{cloth_folder}/{folder_id}-{cloth_id}*cloth*"
            cloth_path_matches = sorted(glob(search))
            logger.debug(f"{search=} found {cloth_path_matches=}")

        assert (
            len(cloth_path_matches) > 0
        ), f"{search=} not found. Try specifying --warp_cloth_dir"

        return cloth_path_matches[0]

    # @overrides(TryonDataset)
    def get_input_cloth_name(self, index):
        cloth_path = self.get_input_cloth_path(index)
        folder_id = VVTDataset.extract_folder_id(cloth_path)
        base_cloth_name = osp.basename(cloth_path)
        frame_name = osp.basename(self.get_person_image_name(index))
        # e.g. 4he21d00f-g11/4he21d00f-g11@10=cloth_front.jpg
        name = osp.join(folder_id, f"{base_cloth_name}.FOR.{frame_name}")
        return name

    ########################
    # PERSON REPRESENTATION
    ########################
    # @overrides(TryonDataset)
    def get_person_image_path(self, index):
        # because we globbed, the path is the list
        return self.image_names[index]

    # @overrides(TryonDataset)
    def get_person_image_name(self, index):
        image_path = self.get_person_image_path(index)
        folder_id = VVTDataset.extract_folder_id(image_path)
        name = osp.join(folder_id, osp.basename(image_path))
        return name

    # @overrides(TryonDataset)
    def get_person_parsed_path(self, index):
        image_path = self.get_person_image_path(index)
        folder = f"{self.opt.datamode}/{self.opt.datamode}_frames_parsing"
        id = VVTDataset.extract_folder_id(image_path)
        parsed_fname = os.path.split(image_path)[-1].replace(".png", "_label.png")
        parsed_path = osp.join(self.root, folder, id, parsed_fname)
        # hacky, if it doesn't exist as _label, then try getting rid of it
        if not os.path.exists(parsed_path):
            parsed_path = parsed_path.replace("_label", "")
        return parsed_path

    # @overrides(TryonDataset)
    def get_person_cocopose_path(self, index):
        image_path = self.get_person_image_path(index)
        folder = f"{self.opt.datamode}/{self.opt.datamode}_frames_keypoint"
        id = VVTDataset.extract_folder_id(image_path)

        keypoint_fname = os.path.split(image_path)[-1].replace(
            ".png", "_keypoints.json"
        )

        pose_path = osp.join(self.root, folder, id, keypoint_fname)
        return pose_path

    # @overrides(TryonDataset)
    def get_person_densepose_path(self, index):
        image_path = self.get_person_image_path(index)
        folder = f"{self.opt.datamode}/densepose"
        id = VVTDataset.extract_folder_id(image_path)

        iuv_fname = os.path.split(image_path)[-1].replace(".png", "_IUV.png")

        densepose_path = osp.join(self.root, folder, id, iuv_fname)
        return densepose_path

    def get_person_flow_path(self, index):
        image_path = self.get_person_image_path(index)
        image_path = image_path.replace(".png", ".flo")
        image_path = image_path.replace(f"{self.opt.datamode}_frames", "optical_flow")
        return image_path

    # @overrides(NFramesInterface)
    def collect_n_frames_indices(self, index):
        """ Walks backwards from the current index to collect self.n_frames_total indices
        before it"""
        indices = []
        # walk backwards to gather frame indices
        for i in range(index, index - self.n_frames_total, -1):
            assert i > -1, "index can't be negative, something's wrong!"
            # if we reach the video boundary, dupe this index for the remaining times
            if i in self._video_start_indices or i == 0:
                num_times = self.n_frames_total - len(indices)
                dupes = [i] * num_times
                indices = dupes + indices  # prepend
                break  # end
            else:
                indices.insert(0, i)
        return indices

    def __len__(self):
        # TODO: make len =
        #  len(self.image_frames) - len(self._video_start_indices) * self.n_frames_total
        #  this will make sure the beginnings of videos are not repeated
        return super().__len__()

    @NFramesInterface.return_n_frames
    def __getitem__(self, index):
        # TODO: if index is at a start index, += it
        return super().__getitem__(index)
