import os.path as osp

from pytorch_lightning import Trainer

from models import find_model_using_name
from options.test_options import TestOptions
from train import main

if __name__ == "__main__":
    main(train=False)
