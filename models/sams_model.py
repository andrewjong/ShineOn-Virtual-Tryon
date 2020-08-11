import argparse
from torch import Tensor
from typing import List, Dict, Tuple

import torch
from torch.nn import L1Loss
from torch.optim import Adam

from models import BaseModel, networks
from models.networks.loss import VGGLoss, GANLoss
from models.networks.sams.sams_generator import SamsGenerator
from options import gan_options
from util import get_prev_data_zero_bounded


class SamsModel(BaseModel):
    """ Self Attentive Multi-Spade """

    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser = super(SamsModel, cls).modify_commandline_options(parser, is_train)
        parser.set_defaults(person_inputs=("agnostic", "densepose"), n_frames=12)
        parser.add_argument(
            "--n_frames_G",
            type=int,
            default=3,
            help="Number of frames to pass into the generator's base at once. "
            "Should be a factor of --n_frames_total.",
        )
        parser.add_argument(
            "--n_frames_D",
            type=int,
            default=4,
            help="Number of frames to pass to multi-scale temporal discriminators. "
            "Should be a factor of --n_frames_total.",
        )
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
        self.n_frames_G = hparams.n_frames_G
        self.n_frames_D = hparams.n_frames_D
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
        synth_output, this_frame_labelmap, gen_frames = self._generate_n_frames(batch)

        input_semantics = torch.cat(tuple(this_frame_labelmap.values()), dim=1)
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

    def _generate_n_frames(self, batch):
        # format: { agnostic: frames, densepose: frames, flow: frames, etc... }
        labelmap: Dict[str, List[Tensor]] = {key: batch[key] for key in self.inputs}

        # make a buffer of previous frames
        gt_shp: Tuple = batch["image"][0].shape
        all_generated_frames: List[Tensor] = [
            torch.zeros(*gt_shp, device=self.device) for _ in range(self.n_frames_total)
        ]

        # generate previous frames before this one
        for fIdx in range(self.n_frames_total):
            # Prepare data...
            # all the guidance for the current frame
            labelmaps_this_frame: Dict[str, Tensor] = {
                map_name: segmap[fIdx] for map_name, segmap in labelmap.items()
            }
            prev_n_frames_G, prev_n_labelmaps = self.get_prev_frames_and_maps(
                batch, fIdx, all_generated_frames
            )
            # forward
            fake_frame: Tensor = self.generator.forward(
                prev_n_frames_G, prev_n_labelmaps, labelmaps_this_frame
            )
            # TODO: INDIVIDUAL DISCRIMINATOR SHOULD CALCULATE LOSS HERE, but we can't :C
            # add to buffer, but don't detach here; temporal discriminator needs it
            all_generated_frames[fIdx] = fake_frame

        return fake_frame, labelmaps_this_frame, all_generated_frames

    def get_prev_frames_and_maps(self, batch, fIdx, all_generated_frames):
        """
        Get previous frames, but protected by zero
        Returns:
            - prev_frames[end_idx - self.n_frames_G]... , prev_frames[end_idx]

        """
        prev_n_frames_G = get_prev_data_zero_bounded(
            all_generated_frames, fIdx, self.n_frames_G
        )
        prev_n_frames_G = [t.detach() for t in prev_n_frames_G]  # detach, easier train

        # The encoder only takes ONE labelmap: "--encoder_input"
        enc_labl_maps = batch[self.hparams.encoder_input]
        prev_n_frames_labelmaps = get_prev_data_zero_bounded(
            enc_labl_maps, fIdx, self.n_frames_G
        )
        return prev_n_frames_G, prev_n_frames_labelmaps

    def _discriminator_step(self, batch):
        ground_truth = batch["image"][-1]
        with torch.no_grad():
            synth_output, this_frame_labelmap, _ = self._generate_n_frames(batch)
            synth_output = synth_output.detach()
            synth_output.requires_grad_()

        input_semantics = torch.cat(tuple(this_frame_labelmap.values()), dim=1)

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
