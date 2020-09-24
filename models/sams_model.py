import argparse
import logging
from typing import Dict, List

import torch
from pytorch_lightning import TrainResult, EvalResult
from torch import Tensor
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data.dataloader import default_collate

from datasets.tryon_dataset import parse_num_channels, TryonDataset
from models.base_model import BaseModel
from models.networks.loss import VGGLoss, GANLoss
from models.networks.sams.sams_generator import SamsGenerator
from .flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d
from options import gan_options
from visualization import tensor_list_for_board

logger = logging.getLogger("logger")


class SamsModel(BaseModel):
    """ Self Attentive Multi-Spade """

    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser = super(SamsModel, cls).modify_commandline_options(parser, is_train)
        parser.set_defaults(person_inputs=("agnostic", "densepose", "flow"))
        parser.add_argument(
            "--encoder_input",
            default="flow",
            help="which of the --person_inputs to use as the encoder segmap input "
            "(only 1 allowed).",
        )
        # num previous frames fed as input = n_frames_total - 1
        parser.set_defaults(n_frames_total=5)
        # batch size effectively becomes n_frames_total * batch
        parser.set_defaults(batch_size=4)
        parser.add_argument(
            "--wt_l1",
            type=float,
            default=1.0,
            help="Weight applied to l1 loss in the generator",
        )
        parser.add_argument(
            "--wt_vgg",
            type=float,
            default=1.0,
            help="Weight applied to vgg loss in the generator",
        )
        parser.add_argument(
            "--wt_multiscale",
            type=float,
            default=1.0,
            help="Weight applied to adversarial multiscale loss in the generator",
        )
        parser.add_argument(
            "--wt_temporal",
            type=float,
            default=1.0,
            help="Weight applied to adversarial temporal loss in the generator",
        )
        parser.add_argument(
            "--norm_D",
            type=str,
            default="spectralinstance",
            help="instance normalization or batch normalization",
        )
        from models import networks

        parser = networks.modify_commandline_options(parser, is_train)
        parser = gan_options.modify_commandline_options(parser, is_train)
        return parser

    @staticmethod
    def apply_default_encoder_input(opt):
        """ Call in Base Options after opt parsed """
        if hasattr(opt, "encoder_input") and opt.encoder_input is None:
            opt.encoder_input = opt.person_inputs[0]
        return opt

    def __init__(self, hparams):
        # Lightning bug, see https://github.com/PyTorchLightning/pytorch-lightning/issues/924#issuecomment-673137383
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        super().__init__(hparams)
        self.n_frames_total = hparams.n_frames_total
        self.n_frames_now = (
            hparams.n_frames_now if hparams.n_frames_now else self.n_frames_total
        )
        self.inputs = hparams.person_inputs + hparams.cloth_inputs
        self.generator = SamsGenerator(hparams)
        self.resample = Resample2d()


        if self.is_train:
            init = hparams.init_type, hparams.init_variance
            self.generator.init_weights(*init)

            from models.networks import MultiscaleDiscriminator

            self.multiscale_discriminator = MultiscaleDiscriminator(hparams)
            self.multiscale_discriminator.init_weights(*init)

            enc_ch = parse_num_channels(hparams.encoder_input)
            temporal_in_channels = self.n_frames_total * (
                enc_ch + TryonDataset.RGB_CHANNELS
            )
            from models.networks import NLayerDiscriminator

            self.temporal_discriminator = NLayerDiscriminator(
                hparams, in_channels=temporal_in_channels
            )
            self.temporal_discriminator.init_weights(*init)

            self.criterion_GAN = GANLoss(hparams.gan_mode)
            self.criterion_l1 = L1Loss()
            self.criterion_VGG = VGGLoss()

            self.wt_l1 = hparams.wt_l1
            self.wt_vgg = hparams.wt_vgg
            self.wt_multiscale = hparams.wt_multiscale
            self.wt_temporal = hparams.wt_temporal

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
            if self.global_step % self.hparams.display_count == 0:
                self.visualize(batch)

        return result

    def validation_step(self, batch, idx) -> Dict[str, Tensor]:
        self.batch = batch
        result = self.generator_step(batch, val=True)
        result.global_step = self.global_step

        return result

    # def validation_epoch_end(self, result) -> EvalResult:
    #     result.val_loss = result["val_loss"].mean()
    #     return result

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        pass

    def generator_step(self, batch, val=False):
        loss_G_adv_multiscale = (  # also calls generator forward
            self.multiscale_adversarial_loss(batch, for_discriminator=False)
            * self.wt_multiscale
        )
        loss_G_adv_temporal = (
            self.temporal_adversarial_loss(batch, for_discriminator=False)
            * self.wt_temporal
        )
        ground_truth = batch["image"][:, -1, :, :, :]
        fake_frame = self.all_gen_frames[:, -1, :, :, :]
        loss_G_l1 = self.criterion_l1(fake_frame, ground_truth) * self.wt_l1
        loss_G_vgg = self.criterion_VGG(fake_frame, ground_truth) * self.wt_vgg

        loss_G = loss_G_l1 + loss_G_vgg + loss_G_adv_multiscale + loss_G_adv_temporal

        # Log
        val_ = "val_" if val else ""
        result = (
            EvalResult(checkpoint_on=loss_G_l1 + loss_G_vgg)
            if val
            else TrainResult(loss_G)
        )
        result.log(f"{val_}loss", loss_G)
        result.log(f"{val_}loss/G/adv_multiscale", loss_G_adv_multiscale, prog_bar=True)
        result.log(f"{val_}loss/G/adv_temporal", loss_G_adv_temporal, prog_bar=True)
        result.log(f"{val_}loss/G/l1+vgg", loss_G_l1 + loss_G_vgg)
        result.log(f"{val_}loss/G/l1", loss_G_l1)
        result.log(f"{val_}loss/G/vgg", loss_G_vgg)
        return result

    def generate_n_frames(self, batch):
        # each Tensor is (b x N-Frames x c x h x w)
        labelmap: Dict[str, Tensor] = {key: batch[key] for key in self.inputs}

        # make a buffer of previous frames, also (b x N x c x h x w)
        all_generated_frames: Tensor = torch.zeros_like(batch["image"])
        flows = torch.unbind(batch["flow"], dim=1) if self.hparams.flow_warp else None

        # generate previous frames before this one.
        #   for progressive training, just generate from here
        start_idx = self.n_frames_total - self.n_frames_now
        for fIdx in range(start_idx, self.n_frames_total):
            # Prepare data...
            # all the guidance for the current frame
            weight_boundary = TryonDataset.RGB_CHANNELS
            labelmaps_this_frame: Dict[str, Tensor] = {
                name: lblmap[:, fIdx, :, :, :] for name, lblmap in labelmap.items()
            }
            prev_n_frames_G, prev_n_labelmaps = self.get_prev_frames_and_maps(
                batch, fIdx, all_generated_frames
            )
            # synthesize
            out: Tensor = self.generator.forward(
                prev_n_frames_G, prev_n_labelmaps, labelmaps_this_frame
            )
            fake_frame = out[:, :weight_boundary, :, :].clone()
            weight_mask = out[:, weight_boundary:, :, :].clone()

            if self.hparams.flow_warp:
                last_generated_frame = all_generated_frames[:, fIdx - 1, :, :, :].clone() if fIdx > 0 else torch.zeros_like(all_generated_frames[:, fIdx, :, :, :])
                warped_flow = self.resample(last_generated_frame, flows[fIdx].contiguous())
                fake_frame = (1 - weight_mask) * warped_flow + weight_mask * fake_frame
            # add to buffer, but don't detach; must go through temporal discriminator
            all_generated_frames[:, fIdx, :, :, :] = fake_frame

        return fake_frame, labelmaps_this_frame, all_generated_frames

    def get_prev_frames_and_maps(self, batch, fIdx, all_G_frames):
        """
        Get previous frames, but padded by zero
        Returns:
            - prev_frames[end_idx - self.n_frames_total]... , prev_frames[end_idx]

        """
        enc_lblmaps: Tensor = batch[self.hparams.encoder_input]
        nframes = self.n_frames_total
        if nframes == 1:
            # (b x 1 x c x h x w)
            prev_n_frames_G = torch.zeros_like(all_G_frames)
            prev_n_labelmaps = torch.zeros_like(enc_lblmaps)
        else:
            n_prev = nframes - 1
            # Previously generated frames.
            # (b x N-1 x c x h x w)
            indices = torch.tensor(
                [(i + 1) % nframes for i in range(fIdx, fIdx + n_prev)],
                device=all_G_frames.device,
            )
            prev_n_frames_G = torch.index_select(all_G_frames, 1, indices).detach()

            # Corresponding encoding maps.
            b, n, c, h, w = enc_lblmaps.shape
            start = n_prev - fIdx  # nframes= 5,fIdx=3
            zero_pad = torch.zeros(b, start, c, h, w).type_as(enc_lblmaps)
            # only up to the last index, which is for current frame
            prev_labelmaps_now = enc_lblmaps[:, start:-1, :, :, :]
            prev_n_labelmaps = torch.cat((zero_pad, prev_labelmaps_now), dim=1)

        return prev_n_frames_G, prev_n_labelmaps

    def multiscale_adversarial_loss(self, batch, for_discriminator):
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

    def temporal_adversarial_loss(self, batch, for_discriminator):
        reals = self.mask_unused_frames(batch["image"])
        b, _, _, h, w = reals.shape
        reals = reals.view(b, -1, h, w)

        # fakes: already prepared by multiscale_discriminator_loss(). is pre-masked
        #  by generate_n_frames
        fakes = self.all_gen_frames.view(b, -1, h, w)

        # enc_labl_maps: the single encoder labelmaps (e.g. flow) for ALL n_frames.
        # for progressive training, should get rid of the extra ones, because generator
        # doesn't see it either
        enc_labl_maps: Tensor = batch[self.hparams.encoder_input]
        enc_labl_maps = self.mask_unused_frames(enc_labl_maps)
        b, _, _, h, w = enc_labl_maps.shape
        input_semantics = enc_labl_maps.view(b, -1, h, w)

        # run through temporal discriminator
        pred_fake, pred_real = self.discriminate(
            self.temporal_discriminator, input_semantics, fakes, reals
        )

        # calculate adversarial loss
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

    def mask_unused_frames(self, tensor: Tensor):
        """ For progressive training, mask out the previous frames.

        Args:
            tensor: (b x Frames x c x h x w) shape.

        Returns:

        """
        n_mask = self.n_frames_total - self.n_frames_now
        b, _, c, h, w = tensor.shape
        zeros_mask = torch.zeros(b, n_mask, c, h, w).type_as(tensor)
        part_to_keep = tensor[:, n_mask:, :, :, :]

        masked_result = torch.cat((zeros_mask, part_to_keep), dim=1)
        return masked_result

    def multiscale_discriminator_step(self, batch):
        loss_D, loss_D_real, loss_D_fake = self.multiscale_adversarial_loss(
            batch, for_discriminator=True
        )

        result = TrainResult(loss_D)
        result.log("loss/D/multi", loss_D, prog_bar=True)
        result.log("loss/D/multi_fake", loss_D_fake)
        result.log("loss/D/multi_real", loss_D_real)
        return result

    def temporal_discriminator_step(self, batch):
        loss_D, loss_D_real, loss_D_fake = self.temporal_adversarial_loss(
            batch, for_discriminator=True
        )

        result = TrainResult(loss_D)
        result.log("loss/D/temporal", loss_D, prog_bar=True)
        result.log("loss/D/temporal_fake", loss_D_fake)
        result.log("loss/D/temporal_real", loss_D_real)
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

    def visualize(self, batch, tag="train"):
        rows = []  # one type per row
        # add inputs
        person_vis_names = self.replace_actual_with_visual()
        for name in person_vis_names:
            # [0] for only the first index in the batch
            frames_list: List[Tensor] = torch.unbind(batch[name], dim=1)
            rows.append(frames_list)
        # add cloths
        cloths_list = torch.unbind(batch["cloth"], dim=1)
        rows.append(cloths_list)
        # add fakes
        fakes_list = torch.unbind(self.all_gen_frames, dim=1)
        rows.append(fakes_list)
        # add reals
        reals_list = torch.unbind(batch["image"], dim=1)
        rows.append(reals_list)
        tensor = tensor_list_for_board(rows)
        # add to experiment
        for i, img in enumerate(tensor):
            self.logger.experiment.add_image(f"{tag}/{i:03d}", img, self.global_step)


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
            if isinstance(p, torch.Tensor):  # single N-Layer Discriminator
                tensor = p
                fake.append(tensor[: tensor.size(0) // 2])
                real.append(tensor[tensor.size(0) // 2 :])
            else:  # multiscale disc, has several N-Layer Discriminators
                fake.append([tensor[: tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2 :] for tensor in p])
    else:
        fake = pred[: pred.size(0) // 2]
        real = pred[pred.size(0) // 2 :]

    return fake, real
