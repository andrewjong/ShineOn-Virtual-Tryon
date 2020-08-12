import argparse
from torch import Tensor
from typing import List, Dict, Tuple

import torch
from torch.nn import L1Loss
from torch.optim import Adam

from models import BaseModel, networks
from models.networks import (
    MultiscaleDiscriminator,
    NLayerDiscriminator,
    parse_num_channels,
    TryonDataset,
)
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
        parser.set_defaults(person_inputs=("agnostic", "densepose", "flow"))
        # num previous frames fed as input = n_frames_total - 1
        parser.set_defaults(n_frames_total=5)
        # batch size effectively becomes n_frames_total * batch
        parser.set_defaults(batch_size=4)
        parser.add_argument(
            "--discriminator",
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
            default="flow",
            help="which of the --person_inputs to use as the encoder segmap input "
            "(only 1 allowed).",
        )
        parser = networks.modify_commandline_options(parser, is_train)
        parser = gan_options.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, hparams):
        super().__init__(hparams)
        self.n_frames_total = hparams.n_frames_total
        self.inputs = hparams.person_inputs + hparams.cloth_inputs
        self.generator = SamsGenerator(hparams)

        if self.isTrain:
            init = hparams.init_type, hparams.init_variance
            self.generator.init_weights(*init)

            self.multiscale_discriminator = MultiscaleDiscriminator(hparams)
            self.multiscale_discriminator.init_weights(*init)

            enc_ch = parse_num_channels(hparams.encoder_input)
            temporal_in_channels = self.n_frames_total * (
                enc_ch + TryonDataset.RGB_CHANNELS
            )
            self.temporal_discriminator = NLayerDiscriminator(
                hparams, in_channels=temporal_in_channels
            )
            self.temporal_discriminator.init_weights(*init)

            self.criterion_GAN = GANLoss(hparams.gan_mode)
            self.criterion_l1 = L1Loss()
            self.criterion_VGG = VGGLoss()
            self.crit_adv_multiscale = None  # TODO
            self.crit_adv_temporal = None  # TODO

    def forward(self, *args, **kwargs):
        self.generator(*args, **kwargs)

    def configure_optimizers(self):
        # must do individual optimizers and schedulers per each network
        optimizer_g = Adam(self.generator.parameters(), self.hparams.lr)
        optimizer_d_multi = Adam(
            self.multiscale_discriminator.parameters(), self.hparams.lr_D
        )
        optimizer_d_temporal = Adam(
            self.temporal_discriminator.parameters(), self.hparams.lr_D
        )
        scheduler_g = self._make_step_scheduler(optimizer_g)
        scheduler_d_multi = self._make_step_scheduler(optimizer_d_multi)
        scheduler_d_temporal = self._make_step_scheduler(optimizer_d_temporal)
        return (
            [optimizer_g, optimizer_d_multi, optimizer_d_temporal],
            [scheduler_g, scheduler_d_multi, scheduler_d_temporal],
        )

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        if optimizer_idx == 0:
            result = self.generator_step(batch)
        elif optimizer_idx == 1:
            result = self.multiscale_discriminator_step(batch)
        else:
            result = self.temporal_discriminator_step(batch)

        return result

    def validation_step(self, batch, idx) -> Dict[str, Tensor]:
        result = self.generator_step(batch)
        log = result["log"]
        val_loss = log["loss_G_l1"] + log["loss_G_vgg"]
        return {"val_loss": val_loss}


    def generator_step(self, batch):
        loss_G_adv_multiscale = self.multiscale_discriminator_loss(
            batch, for_discriminator=False
        )
        loss_G_adv_temporal = self.temporal_discriminator_loss(
            batch, for_discriminator=False
        )
        ground_truth = batch["image"][:, -1, :, :, :]
        fake_frame = self.all_gen_frames[:, -1, :, :, :]
        loss_G_l1 = self.criterion_l1(fake_frame, ground_truth)
        loss_G_vgg = self.criterion_VGG(fake_frame, ground_truth)

        loss_G = loss_G_l1 + loss_G_vgg + loss_G_adv_multiscale + loss_G_adv_temporal

        # Log
        log = {
            "loss_G": loss_G,
            "loss_G_adv_multiscale": loss_G_adv_multiscale,
            "loss_G_adv_temporal": loss_G_adv_temporal,
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
        # each Tensor is (b x N-Frames x c x h x w)
        labelmap: Dict[str, Tensor] = {key: batch[key] for key in self.inputs}

        # make a buffer of previous frames, also (b x N x c x h x w)
        all_generated_frames: Tensor = torch.zeros_like(batch["image"])

        # generate previous frames before this one
        for fIdx in range(self.n_frames_total):
            # Prepare data...
            # all the guidance for the current frame
            labelmaps_this_frame: Dict[str, Tensor] = {
                name: lblmap[:, fIdx, :, :, :] for name, lblmap in labelmap.items()
            }
            prev_n_frames_G, prev_n_labelmaps = self.get_prev_frames_and_maps(
                batch, fIdx, all_generated_frames
            )
            # forward
            fake_frame: Tensor = self.generator.forward(
                prev_n_frames_G, prev_n_labelmaps, labelmaps_this_frame
            )
            # add to buffer, but don't detach here; must go through temporal discriminator
            all_generated_frames[:, fIdx, :, :, :] = fake_frame

        return fake_frame, labelmaps_this_frame, all_generated_frames

    def get_prev_frames_and_maps(self, batch, fIdx, all_G_frames):
        """
        Get previous frames, but protected by zero
        Returns:
            - prev_frames[end_idx - self.n_frames_total]... , prev_frames[end_idx]

        """
        enc_lblmaps: Tensor = batch[self.hparams.encoder_input]
        n = self.hparams.n_frames_total
        if n == 1:
            # (b x 1 x c x h x w)
            prev_n_frames_G = torch.zeros_like(all_G_frames)
            prev_n_label_maps = torch.zeros_like(enc_lblmaps)
        else:
            # (b x N-1 x c x h x w)
            indices = torch.tensor(
                [(i + 1) % n for i in range(fIdx, fIdx + n - 1)],
                device=all_G_frames.device,
            )
            prev_n_frames_G = torch.index_select(all_G_frames, 1, indices).detach()
            prev_n_label_maps = torch.index_select(enc_lblmaps, 1, indices).detach()

        return prev_n_frames_G, prev_n_label_maps

    def multiscale_discriminator_loss(self, batch, for_discriminator):
        # Forward
        if not for_discriminator:  # generator
            fake_frame, labelmaps_this_frame, all_gen_frames = self.generate_n_frames(
                batch
            )
            self.all_gen_frames = all_gen_frames  # save for temporal discriminator loss
        else:  # discriminator, we detach from the generator
            with torch.no_grad():
                # generate fresh, so that discriminator works on latest generator
                (
                    fake_frame,
                    labelmaps_this_frame,
                    all_gen_frames,
                ) = self.generate_n_frames(batch)
                fake_frame = fake_frame.detach().requires_grad_()
                # save this for temporal discriminator
                self.all_gen_frames = all_gen_frames.detach().requires_grad_()
        # LOSSES
        # Multiscale adversarial
        input_semantics = torch.cat(tuple(labelmaps_this_frame.values()), dim=1)
        ground_truth = batch["image"][:, -1, :, :, :]
        pred_fake, pred_real = self.discriminate(
            self.multiscale_discriminator, input_semantics, fake_frame, ground_truth
        )
        loss_real = self.criterion_GAN(
            pred_real, True, for_discriminator=for_discriminator
        )
        if not for_discriminator:
            return loss_real
        else:
            loss_fake = self.criterion_GAN(
                pred_fake, False, for_discriminator=for_discriminator
            )
            loss = (loss_fake + loss_real) / 2
            return loss, loss_real, loss_fake

    def temporal_discriminator_loss(self, batch, for_discriminator):
        ground_truth = batch["image"]  # this time it's all of them
        b, n, c, h, w = ground_truth.shape
        reals = ground_truth.view(b, n * c, h, w)
        # already prepared by multiscale_discriminator_loss(). that one needs to be
        # called first!
        fakes = self.all_gen_frames.view(b, n * c, h, w)

        enc_labl_maps: Tensor = batch[self.hparams.encoder_input]
        input_semantics = enc_labl_maps.view(b, -1, h, w)
        pred_fake, pred_real = self.discriminate(
            self.temporal_discriminator, input_semantics, fakes, reals
        )

        loss_real = self.criterion_GAN(
            pred_real, True, for_discriminator=for_discriminator
        )
        if not for_discriminator:
            return loss_real
        else:
            loss_fake = self.criterion_GAN(
                pred_fake, False, for_discriminator=for_discriminator
            )
            loss = (loss_fake + loss_real) / 2
            return loss, loss_real, loss_fake

    def multiscale_discriminator_step(self, batch):
        loss_D, loss_D_real, loss_D_fake = self.multiscale_discriminator_loss(
            batch, for_discriminator=True
        )

        log = {
            "loss_D_multi": loss_D,
            "loss_D_multi_fake": loss_D_fake,
            "loss_D_multi_real": loss_D_real,
        }
        result = {
            "loss": loss_D,
            "log": log,
            "progress_bar": log,
        }
        return result

    def temporal_discriminator_step(self, batch):
        loss_D, loss_D_real, loss_D_fake = self.temporal_discriminator_loss(
            batch, for_discriminator=True
        )

        log = {
            "loss_D_temporal": loss_D,
            "loss_D_temporal_fake": loss_D_fake,
            "loss_D_temporal_real": loss_D_real,
        }
        result = {
            "loss": loss_D,
            "log": log,
            "progress_bar": log,
        }
        return result

    def discriminate(self, discriminator, input_semantics, fake_image, real_image):
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

        discriminator_out = discriminator(fake_and_real)

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
