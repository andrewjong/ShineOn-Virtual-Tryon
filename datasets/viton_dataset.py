# from overrides import overrides

from datasets import CpVtonDataset, CPDataLoader
import os.path as osp


class VitonDataset(CpVtonDataset):
    """ CP-VTON dataset with the original Viton folder structure """
    def __init__(self, opt):
        super().__init__(opt)

    #@overrides
    def load_file_paths(self):
        """
        Reads the datalist txt file for CP-VTON
        sets self.image_names and self.cloth_names. they should correspond 1-to-1
        """
        im_names = []
        c_names = []
        with open(osp.join(self.opt.dataroot, self.opt.data_list), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.image_names = im_names
        self.cloth_names = c_names


    ########################
    # CLOTH REPRESENTATION
    ########################
    #@overrides
    def get_input_cloth_path(self, index):
        """
        Get the file path for the product image input.
        Called by get_input_cloth()
        """
        c_name = self.get_input_cloth_name(index)
        folder = "cloth" if self.stage == "GMM" else "warp-cloth"
        cloth_path = osp.join(self.data_path, folder, c_name)
        return cloth_path

    #@overrides
    def get_input_cloth_name(self, index):
        # determines the written thing
        return self.cloth_names[index]

    ########################
    # PERSON REPRESENTATION
    ########################

    #@overrides
    def get_person_image_name(self, index):
        """ basename of the image file """
        return self.image_names[index]

    #@overrides
    def get_person_image_path(self, index):
        im_name = self.get_person_image_name(index)
        image_path = osp.join(self.data_path, "image", im_name)
        return image_path

    #@overrides
    def get_person_parsed_path(self, index):
        """ path of the clothing seguemtnation """
        im_name = self.get_person_image_name(index)
        parse_name = im_name.replace(".jpg", ".png")
        parsed_path = osp.join(self.data_path, "image-parse", parse_name)
        return parsed_path

    #@overrides
    def get_input_person_pose_path(self, index):
        """ path to pose keypoints """
        im_name = self.get_person_image_name(index)
        _pose_name = im_name.replace(".jpg", "_keypoints.json")
        pose_path = osp.join(self.data_path, "pose", _pose_name)
        return pose_path


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--shuffle", action="store_true", help="shuffle input data")
    parser.add_argument("-b", "--batch-size", type=int, default=4)
    parser.add_argument("-j", "--workers", type=int, default=1)

    opt = parser.parse_args()
    dataset = VitonDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print(
        "Size of the dataset: %05d, dataloader: %04d"
        % (len(dataset), len(data_loader.data_loader))
    )
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed

    embed()
