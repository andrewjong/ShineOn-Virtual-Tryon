import numpy as np
import torch
from torch import nn
import sys
from torchvision import transforms
import os
#from models.base_model import BaseModel
from PIL import Image
import matplotlib.pyplot as plt
#from flownet2_pytorch.utils.flow_utils import visulize_flow_file
class FlowNet(nn.Module):

    def __init__(self):
        super(FlowNet, self).__init__()
        #BaseModel.initialize(self, opt)

        # flownet 2
        from .flownet2_pytorch import models as flownet2_models
        from .flownet2_pytorch.utils import tools as flownet2_tools
        from .flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d

        self.flowNet = flownet2_tools.module_to_dict(flownet2_models)['FlowNet2']().cuda()#self.gpu_ids[0])
        checkpoint = torch.load('models/flownet2_pytorch/FlowNet2_checkpoint.pth.tar')
        self.flowNet.load_state_dict(checkpoint['state_dict'])
        self.flowNet.eval()
        self.resample = Resample2d()
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input_A, input_B):
        with torch.no_grad():
            size = input_A.size()
            assert (len(size) == 4 or len(size) == 5)
            if len(size) == 5:
                b, n, c, h, w = size
                input_A = input_A.view(-1, c, h, w)
                input_B = input_B.view(-1, c, h, w)
                flow, conf = self.compute_flow_and_conf(input_A, input_B)
                return flow.view(b, n, 2, h, w), conf.view(b, n, 1, h, w)
            else:
                return self.compute_flow_and_conf(input_A, input_B)

    def compute_flow_and_conf(self, im1, im2):
        assert (im1.size()[1] == 3)
        assert (im1.size() == im2.size())
        old_h, old_w = im1.size()[2], im1.size()[3]
        new_h, new_w = old_h // 64 * 64, old_w // 64 * 64
        if old_h != new_h:
            downsample = torch.nn.Upsample(size=(new_h, new_w), mode='bilinear')
            upsample = torch.nn.Upsample(size=(old_h, old_w), mode='bilinear')
            im1 = downsample(im1)
            im2 = downsample(im2)
        self.flowNet.cuda(im1.get_device())
        data1 = torch.cat([im1.unsqueeze(2), im2.unsqueeze(2)], dim=2)
        flow1 = self.flowNet(data1)
        conf = (self.norm(im1 - self.resample(im2, flow1)) < 0.02).float()
        if old_h != new_h:
            flow1 = upsample(flow1) * old_h / new_h
            conf = upsample(conf)
        return flow1.detach(), conf.detach()

    def norm(self, t):
        return torch.sum(t * t, dim=1, keepdim=True)

"""f = FlowNet()
to_tensor_and_norm_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
im1 = '/data_hdd/fw_gan_vvt/test/test_frames/4he21d00f-g11/1500448501083_frame_163.png'
im2 = '/data_hdd/fw_gan_vvt/test/test_frames/4he21d00f-g11/1500448501083_frame_164.png'
assert os.path.exists(im1)
assert os.path.exists(im2)
im1 = Image.open(im1)
im2 = Image.open(im2)
im1 = to_tensor_and_norm_rgb(im1).unsqueeze(0).cuda()
im2 = to_tensor_and_norm_rgb(im2).unsqueeze(0).cuda()
flow_, conf_ = f(im1, im2)
print(flow_.size())
print(conf_.size())
#flow_ = flow_.view(256, 192, 2).cpu().numpy().astype(np.float32)
#conf_ = conf_.view(256, 192, 1).cpu().numpy().astype(np.float32)

#plt.imsave(flow_[:,:,0], "output_flow1.jpg")
#plt.imsave(flow_[:,:,1], "output_flow2.jpg")
#plt.imsave(conf_, "output_conf.jpg")
#torchvision.utils.save_image(flow_[0], "output_flow.jpg")
#torchvision.utils.save_image(conf_, "output_conf.jpg")




# save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close()


im1 = flow_.squeeze().data.cpu().numpy().transpose(1, 2, 0)
#im2 = conf_.squeeze().data.cpu().numpy().transpose(1, 2, 0)
writeFlow("im1.flo", im1)
visulize_flow_file("im1.flo", "./")
#writeFlow("im2.flo", im2)"""
