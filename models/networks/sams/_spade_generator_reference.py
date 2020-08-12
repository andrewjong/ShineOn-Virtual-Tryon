import torch
from torch import nn as nn
from torch.nn import functional as F

from models.networks.sams.spade import AnySpadeResBlock


class SPADEGenerator(BaseNetwork):
    """ For reference only """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G="spectralspadesyncbatch3x3")
        parser.add_argument(
            "--num_upsampling_layers",
            choices=("normal", "more", "most"),
            default="normal",
            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator",
        )

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        num_feat = opt.ngf

        self.start_width, self.start_height = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(
                opt.z_dim, 16 * num_feat * self.start_width * self.start_height
            )
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * num_feat, 3, padding=1)

        self.head_0 = AnySpadeResBlock(16 * num_feat, 16 * num_feat, opt.norm_G, TODO, TODO)

        self.G_middle_0 = AnySpadeResBlock(16 * num_feat, 16 * num_feat, opt.norm_G, TODO, TODO)
        self.G_middle_1 = AnySpadeResBlock(16 * num_feat, 16 * num_feat, opt.norm_G, TODO, TODO)

        self.up_0 = AnySpadeResBlock(16 * num_feat, 8 * num_feat, opt.norm_G, TODO, TODO)
        self.up_1 = AnySpadeResBlock(8 * num_feat, 4 * num_feat, opt.norm_G, TODO, TODO)
        self.up_2 = AnySpadeResBlock(4 * num_feat, 2 * num_feat, opt.norm_G, TODO, TODO)
        self.up_3 = AnySpadeResBlock(2 * num_feat, 1 * num_feat, opt.norm_G, TODO, TODO)

        final_nc = num_feat

        if opt.num_upsampling_layers == "most":
            self.up_4 = AnySpadeResBlock(1 * num_feat, num_feat // 2, opt.norm_G, TODO, TODO)
            final_nc = num_feat // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == "normal":
            num_up_layers = 5
        elif opt.num_upsampling_layers == "more":
            num_up_layers = 6
        elif opt.num_upsampling_layers == "most":
            num_up_layers = 7
        else:
            raise ValueError(
                "opt.num_upsampling_layers [%s] not recognized"
                % opt.num_upsampling_layers
            )

        start_width = opt.crop_size // (2 ** num_up_layers)
        start_height = round(start_width / opt.aspect_ratio)

        return start_width, start_height

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(
                    input.size(0),
                    self.opt.z_dim,
                    dtype=torch.float32,
                    device=input.get_device(),
                )
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.start_height, self.start_width)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.start_height, self.start_width))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)


        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)


        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x