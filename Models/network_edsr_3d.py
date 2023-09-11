# Author: Qinyang Shou
# qinyangs@usc.edu

import torch.nn as nn
import torch

# need to test the input and output shape before and after the convolution
class ResidualBlock3D(nn.Module):
    def __init__(self, filter_size, act_type, scale_factor, blk_num):
        super(ResidualBlock3D, self).__init__()
        # conv
        self.add_module("resblock_%i_conv_1" % blk_num, nn.Conv3d(filter_size, filter_size, (3,3,3), padding='same'))
        #  activation
        if act_type == "LeakyReLu":
            self.add_module('resblock_%i_act' % blk_num, nn.LeakyReLU(0.2))
        else:
            self.add_module('resblock_%i_act' % blk_num, nn.ReLU(inplace=True))

        # conv
        self.add_module("resblock_%i_conv_2" % blk_num, nn.Conv3d(filter_size, filter_size, (3,3,3), padding='same'))
        # constMultiplier
        # add
        # self.k = nn.Parameter(torch.tensor(scale_factor), requires_grad=True)
        self.blk_num = blk_num

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)
        assert not torch.equal(out, x)
        out *= 0.1
        out += x
        return out

# ResNet
class edsr_3D(nn.Module):
    def __init__(self, img_channel=1, layers = 32, features = 64, act_type ='relu', scale_factor = 0.1):
        super(edsr_3D, self).__init__()
        # here the img_channels should be different modalities
        self.in_channels = img_channel

        self.add_module("conv0", nn.Conv3d(1, features, (3,3,3), padding='same'))

        self.block_layer = self.make_layer(features, act_type, scale_factor, layers)
        self.add_module("conv_penultimate", nn.Conv3d(features, features, (3,3,3), padding='same'))
        self.add_module("conv_final", nn.Conv3d(features, 1, (3,3,3), padding='same'))
        #self.add_module("Flatten", nn.Flatten(0,1))
        #self.add_module("conv3D_2D",  nn.Conv2d(img_channel,1,(3,3), padding='same'))
        
    def make_layer(self, features, act_type, scale_factor, blk_num):

        layers = []

        for i in range(blk_num):
            layers.append(ResidualBlock3D(features, act_type, scale_factor, i))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv0(x)
        conv_1 = out
        out = self.block_layer(out)
        out = self.conv_penultimate(out)
        out += conv_1
        out = self.conv_final(out)
        #out = self.Flatten(out)
        #out = self.conv3D_2D(out)
        return out

my_EDSR_3D = edsr_3D(layers=20, features=64)
print(sum(p.numel() for p in my_EDSR_3D.parameters()))
# x = torch.zeros((5,1,96,96,48))
# y = my_EDSR_3D(x)
# print(y.shape)
