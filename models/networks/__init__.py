""" A lot of this code credit to NVIDIA """
from torch.nn import init

from models.networks.base_network import BaseNetwork
from models.networks.discriminator import *
from models.networks.loss import *
from models.networks.sams.sams_generator import SamsGenerator


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = "models.networks." + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), (
        "Class %s should be a subclass of BaseNetwork" % network
    )

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    parser = SamsGenerator.modify_commandline_options(parser, is_train)
    if is_train:
        for d in ["multiscale"]:
            netD_cls = find_network_using_name(d, "discriminator")
            parser = netD_cls.modify_commandline_options(parser, is_train)
    # netE_cls = find_network_using_name("conv", "encoder")
    # parser = netE_cls.modify_commandline_options(parser, is_train)

    return parser


def create_network(cls, opt):
    net = cls(opt)
    # net.print_network()
    if len(opt.gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net


def define_D(name, opt):
    netD_cls = find_network_using_name(name, "discriminator")
    return create_network(netD_cls, opt)


# weights init code from CPVton
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find("Linear") != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type="normal"):
    print("initialization method [%s]" % init_type)
    if init_type == "normal":
        net.apply(weights_init_normal)
    elif init_type == "xavier":
        net.apply(weights_init_xavier)
    elif init_type == "kaiming":
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError(
            "initialization method [%s] is not implemented" % init_type
        )
