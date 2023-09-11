import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, filter_size, act_type, scale_factor, blk_num):
        super(ResidualBlock, self).__init__()
        # conv
        self.add_module("resblock_%i_conv_1" % blk_num, nn.Conv2d(filter_size, filter_size, (3,3), padding='same'))
        #  activation
        if act_type == "LeakyReLu":
            self.add_module('resblock_%i_act' % blk_num, nn.LeakyReLU(0.2))
        else:
            self.add_module('resblock_%i_act' % blk_num, nn.ReLU(inplace=True))

        # conv
        self.add_module("resblock_%i_conv_2" % blk_num, nn.Conv2d(filter_size, filter_size, (3,3), padding='same'))
        self.blk_num = blk_num

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)
        assert not torch.equal(out, x)
        out *= 0.1 # scale the output
        out += x
        return out

# ResNet
class edsr_hr(nn.Module):
    def __init__(self, in_channels=1, layers = 32, features = 64, act_type ='relu', scale_factor = 0.1, global_conn = False, main_channel = 1):
        super(edsr_hr, self).__init__()
        self.in_channels = in_channels
        self.main_channel = main_channel
        self.global_conn = global_conn
        self.add_module("conv0", nn.Conv2d(self.in_channels, features, (3,3), padding='same'))

        self.block_layer = self.make_layer(features, act_type, scale_factor, layers)
        self.add_module("conv_penultimate", nn.Conv2d(features, features, (3,3), padding='same'))
        self.add_module("conv_final", nn.Conv2d(features, 1, (3,3), padding='same'))
    
    def make_layer(self, features, act_type, scale_factor, blk_num):

        layers = []

        for i in range(blk_num):
            layers.append(ResidualBlock(features, act_type, scale_factor, i))
        return nn.Sequential(*layers)

    def forward(self, x):
        # for test purpose only
        if self.global_conn:
            out0 = x[:,self.main_channel:self.main_channel+1,:,:]
        #
        out = self.conv0(x)
        conv_1 = out
        out = self.block_layer(out)
        out = self.conv_penultimate(out)
        out += conv_1
        out = self.conv_final(out)
        if self.global_conn:
            out = out + out0
            
        return out

# test
'''
my_EDSR = edsr_hr(in_channels=3, layers=20)
print(sum(p.numel() for p in my_EDSR.parameters()))
'''
