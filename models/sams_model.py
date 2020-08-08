import argparse
from torch import Tensor
from typing import List, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.nn import L1Loss

from models import BaseModel
from models.networks.loss import VGGLoss, GANLoss
from models.networks.sams.sams_generator import SamsGenerator


""" Self Attentive Multi-Spade """


class SamsModel(BaseModel):
    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser = super(SamsModel, cls).modify_commandline_options(parser, is_train)
        parser.set_defaults(inputs=("agnostic", "cloth", "densepose", "flow"))
        parser.add_argument(
            "--gan_mode", default="hinge", choices=GANLoss.AVAILABLE_MODES
        )
        parser.add_argument(
            "--netD", default="multiscale", choices=("multiscale", "nlayer")
        )
        return parser

    def __init__(self, hparams):
        super().__init__(hparams)
        self.generator = SamsGenerator(hparams)

        self.netD = None  # TODO
        self.temporal_discriminator = None  # TODO

        self.criterion_gan = GANLoss(hparams.gan_mode)
        self.criterion_l1 = L1Loss()
        self.criterion_vgg = VGGLoss()
        self.crit_adv_multiscale = None  # TODO
        self.crit_adv_temporal = None  # TODO

    def forward(self, *args, **kwargs):
        self.generator(*args, **kwargs)

    def configure_optimizers(self):
        # must do individual optimizers and schedulers per each network
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0:
            result = self._generator_step(batch, batch_idx)
        else:
            result = self._discriminator_step(batch, batch)
            pass
            # discriminator, remember to update discriminator slower
            # disc_0_outputs = self.netD(batch)
            # disc_1_outputs = self.temporal_discriminator(batch)

        return result

    def _generator_step(self, batch, batch_idx):
        # List of: [ agnostic_frames, densepose_frames, flow_frames ].
        #  each is of length "n_frames"
        segmaps: Dict[str : List[Tensor]] = {
            key: batch[key] for key in self.hparams.inputs
        }
        # make a buffer of previous frames
        frame_shape: Tuple = segmaps["image"][0].shape
        prev_frames: List[Tensor] = [
            torch.zeros(*frame_shape, device=self.device) for _ in range(self.n_frames)
        ]
        agnostic_shape: Tuple = segmaps["agnostic"][0].shape
        # generate previous frames before this one
        for frame_idx in range(self.n_frames):
            # prepare data
            this_frame_segmaps: Dict[str:Tensor] = {
                key: segmap[frame_idx] for key, segmap in segmaps.items()
            }
            prev_frame_agnostics = [
                batch["agnostic"][i] for i in range(0, frame_idx)
            ] + [
                torch.zeros(*agnostic_shape, device=self.device)
                for _ in range(frame_idx, self.n_frames)
            ]
            # forward
            synth_output: Tensor = self.generator.forward(
                prev_frames, prev_frame_agnostics, this_frame_segmaps
            )
            # comment: should we detach()? Ziwei says yes, easier to train
            prev_frames[frame_idx] = synth_output.detach()

        # loss
        ground_truth = batch["image"][-1]
        loss_gan = self.criterion_gan()
        loss_l1 = self.criterion_l1(synth_output, ground_truth)
        loss_vgg = self.criterion_vgg(synth_output, ground_truth)
        # TODO: ADVERSARIAL LOSSES

        loss = loss_gan + loss_l1 + loss_vgg

        result = {"loss": loss}
        return result

    def _discriminator_step(self, batch, batch_idx, optimizer_idx):
        result = {"loss": 0}
        return result

    def discriminate(self, input_semantics, fake_image, real_image):
        """
        Given fake and real image, return the prediction of discriminator
        for each fake and real image.
        """
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = divide_pred(discriminator_out)

        return pred_fake, pred_real


def divide_pred(pred):
    """
    Take the prediction of fake and real images from the combined batch
    """
    # the prediction contains the intermediate outputs of multiscale GAN,
    # so it's usually a list
    if type(pred) == list:
        fake = []
        real = []
        for p in pred:
            fake.append([tensor[: tensor.size(0) // 2] for tensor in p])
            real.append([tensor[tensor.size(0) // 2 :] for tensor in p])
    else:
        fake = pred[: pred.size(0) // 2]
        real = pred[pred.size(0) // 2 :]

    return fake, real
