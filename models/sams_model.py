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


class SamsModel(BaseModel):
    """ Self Attentive Multi-Spade """

    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser = super(SamsModel, cls).modify_commandline_options(parser, is_train)
        parser.set_defaults(person_inputs=("agnostic", "densepose"))
        parser.add_argument(
            "--gan_mode", default="hinge", choices=GANLoss.AVAILABLE_MODES
        )
        parser.add_argument(
            "--netD", default="multiscale", choices=("multiscale", "nlayer")
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
        return parser

    def __init__(self, hparams):
        super().__init__(hparams)
        self.inputs = hparams.person_inputs + hparams.cloth_inputs
        self.generator = SamsGenerator(hparams)

        if self.isTrain:
            self.netD = networks.define_D(hparams)
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
        optimizer_g = Adam(self.generator.parameters(), self.hparams.lr)
        optimizer_d = Adam(self.netD.parameters(), self.hparams.lr)
        scheduler_g = self._make_step_scheduler(optimizer_g)
        scheduler_d = self._make_step_scheduler(optimizer_d)
        return [optimizer_g, optimizer_g], [scheduler_g, scheduler_d]

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0:
            result = self._generator_step(batch)
        else:
            result = self._discriminator_step(batch)
            pass
            # discriminator, remember to update discriminator slower
            # disc_0_outputs = self.netD(batch)
            # disc_1_outputs = self.temporal_discriminator(batch)

        return result

    def _generator_step(self, batch):
        # format: { agnostic: frames, densepose: frames, flow: frames, etc... }
        segmaps: Dict[str, List[Tensor]] = {key: batch[key] for key in self.inputs}
        # make a buffer of previous frames
        ground_truth = batch["image"][-1]
        gt_shape: Tuple = ground_truth.shape
        prev_frames: List[Tensor] = [
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
                prev_frames, prev_frame_encoder_maps, this_frame_segmaps
            )
            # add to buffer
            # comment: should we detach()? Ziwei says yes, easier to train
            prev_frames[frame_idx] = synth_output.detach()

        input_semantics = torch.cat(tuple(this_frame_segmaps.values()), dim=1)
        pred_fake, pred_real = self.discriminate(
            input_semantics, synth_output, ground_truth
        )

        # loss
        loss_gan = self.criterionGAN(pred_fake, True, for_discriminator=False)
        loss_l1 = self.criterion_l1(synth_output, ground_truth)
        loss_vgg = self.criterion_vgg(synth_output, ground_truth)

        loss = loss_gan + loss_l1 + loss_vgg

        # Log
        log = {
            "loss": loss,
            "loss_gan": loss_gan,
            "loss_l1": loss_l1,
            "loss_vgg": loss_vgg,
        }
        result = {"loss": loss, "log": log, "progress_bar": log}
        return result

    def _discriminator_step(self, batch):
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

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        pass


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
