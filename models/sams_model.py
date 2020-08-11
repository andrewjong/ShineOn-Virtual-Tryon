import argparse
from torch import Tensor
from typing import List, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.nn import L1Loss
from torch.optim import Adam

from models import BaseModel, networks
from models.networks.loss import VGGLoss, GANLoss
from models.networks.sams.sams_generator import SamsGenerator
from options import gan_options
from util import without_key


class SamsModel(BaseModel):
    """ Self Attentive Multi-Spade """

    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser = super(SamsModel, cls).modify_commandline_options(parser, is_train)
        parser.set_defaults(person_inputs=("agnostic", "densepose"), n_frames=3)
        parser.add_argument(
            "--netD",
            nargs="+",
            default=("multiscale", "temporal"),
            choices=("multiscale", "temporal", "nlayer"),
        )
        parser.add_argument(
            "--norm_D",
            type=str,
            default="spectralinstance",
            help="instance normalization or batch normalization",
        )
        parser.add_argument(
            "--encoder_input",
            help="which of the --person_inputs to use as the encoder segmap input "
            "(only 1 allowed). Defaults to the first alphabetically in "
            "--person_inputs",
        )
        parser = networks.modify_commandline_options(parser, is_train)
        parser = gan_options.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, hparams):
        super().__init__(hparams)
        self.inputs = hparams.person_inputs + hparams.cloth_inputs
        self.generator = SamsGenerator(hparams)

        if self.isTrain:
            self.netD = networks.define_D("multiscale", hparams)
            # self.temporal_discriminator = networks.define_D("temporal", hparams)

            self.criterion_gan = GANLoss(hparams.gan_mode)
            self.criterion_l1 = L1Loss()
            self.criterion_vgg = VGGLoss()
            self.crit_adv_multiscale = None  # TODO
            self.crit_adv_temporal = None  # TODO

    def forward(self, *args, **kwargs):
        self.generator(*args, **kwargs)

    def configure_optimizers(self):
        # must do individual optimizers and schedulers per each network
        optimizer_g = Adam(self.generator.parameters(), self.hparams.lr)
        optimizer_d = Adam(self.netD.parameters(), self.hparams.lr)
        scheduler_g = self._make_step_scheduler(optimizer_g)
        scheduler_d = self._make_step_scheduler(optimizer_d)
        return [optimizer_g, optimizer_g], [scheduler_g, scheduler_d]

    def training_step(self, batch, batch_idx, optimizer_idx):

        if True or optimizer_idx == 0:
            result = self._generator_step(batch)
        else:
            result = self._discriminator_step(batch)

        return result

    def _generator_step(self, batch):
        ground_truth = batch["image"][-1]
        synth_output, this_frame_segmaps, gen_frames = self.generate_n_frames(batch)

        input_semantics = torch.cat(tuple(this_frame_segmaps.values()), dim=1)
        pred_fake, pred_real = self.discriminate(
            input_semantics, synth_output, ground_truth
        )

        # loss_G
        loss_G_gan = self.criterion_gan(pred_fake, True, for_discriminator=False)
        loss_G_l1 = self.criterion_l1(synth_output, ground_truth)
        loss_G_vgg = self.criterion_vgg(synth_output, ground_truth)

        loss_G = loss_G_gan + loss_G_l1 + loss_G_vgg

        # Log
        log = {
            "loss_G": loss_G,
            "loss_G_gan": loss_G_gan,
            "loss_G_l1": loss_G_l1,
            "loss_G_vgg": loss_G_vgg,
        }
        result = {
            "loss": loss_G,
            "log": log,
            "progress_bar": log,
        }
        return result

    def generate_n_frames(self, batch):
        # format: { agnostic: frames, densepose: frames, flow: frames, etc... }
        segmaps: Dict[str, List[Tensor]] = {key: batch[key] for key in self.inputs}
        # make a buffer of previous frames
        ground_truth = batch["image"][-1]
        gt_shape: Tuple = ground_truth.shape
        generated_frames: List[Tensor] = [
            torch.zeros(*gt_shape, device=self.device) for _ in range(self.n_frames)
        ]
        encoder_maps_shape: Tuple = segmaps[self.hparams.encoder_input][0].shape
        # generate previous frames before this one
        for frame_idx in range(self.n_frames):
            # Prepare data...
            # all the guidance for the current frame
            this_frame_segmaps: Dict[str, Tensor] = {
                key: segmap[frame_idx] for key, segmap in segmaps.items()
            }
            # just the encoder maps for the previous frames
            prev_frame_encoder_maps = [
                batch[self.hparams.encoder_input][i] for i in range(0, frame_idx)
            ] + [
                torch.zeros(*encoder_maps_shape, device=self.device)
                for _ in range(frame_idx, self.n_frames)
            ]
            # forward
            synth_output: Tensor = self.generator.forward(
                generated_frames, prev_frame_encoder_maps, this_frame_segmaps
            )
            # add to buffer
            # comment: should we detach()? Ziwei says yes, easier to train
            generated_frames[frame_idx] = synth_output  # .detach()

        return synth_output, this_frame_segmaps, generated_frames

    def _discriminator_step(self, batch):
        ground_truth = batch["image"][-1]
        with torch.no_grad():
            synth_output, this_frame_segmaps, _ = self.generate_n_frames(batch)
            synth_output = synth_output.detach()
            synth_output.requires_grad_()

        input_semantics = torch.cat(tuple(this_frame_segmaps.values()), dim=1)

        pred_fake, pred_real = self.discriminate(
            input_semantics, synth_output, ground_truth
        )
        # TODO: TEMPORAL DISCRIMINATOR

        loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)
        loss_D = loss_D_fake + loss_D_real

        log = {"loss_D": loss_D, "loss_D_fake": loss_D_fake, "loss_D_real": loss_D_real}
        result = {
            "loss": loss_D,
            "log": log,
            "progress_bar": log,
        }
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

        pred_fake, pred_real = split_predictions(discriminator_out)

        return pred_fake, pred_real

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        pass


def split_predictions(pred):
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
