import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, in_channel = 32, out_channel_1 = 128, out_channel_2 = 32, dilation = 1):
        super(ResBlock, self).__init__()
        # conv
        self.add_module("resblock_conv_1", nn.Conv2d(in_channel, out_channel_1, (3,3), dilation = dilation, padding='same'))
        self.add_module('resblock_act', nn.ReLU(inplace=True))
        self.add_module("resblock_conv_2", nn.Conv2d(out_channel_1, out_channel_2, (3,3), padding='same'))

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)
        out += x
        return out

class DWAN_network(nn.Module):
    def __init__(self, img_channel=1):
        super(DWAN_network, self).__init__()
        self.in_channels = img_channel
        self.add_module("conv0", nn.Conv2d(self.in_channels, 32, (3,3), padding='same'))
        self.local_pathway = self.make_local_pathway()
        self.global_pathway = self.make_global_pathway()
        self.add_module("conv_final", nn.Conv2d(64, 1, (3,3), padding='same'))
        self.add_module("direct_conv",nn.Conv2d(self.in_channels,1,(3,3), padding = 'same'))
    
    def make_local_pathway(self):

        layers = []
        # 4 layers of resblock in local pathway
        for i in range(4):
            layers.append(ResBlock(in_channel = 32, out_channel_1 = 128, out_channel_2=32,dilation=1))
        return nn.Sequential(*layers)
    
    def make_global_pathway(self):
        layers = []
        for i in range(4):
            layers.append(ResBlock(in_channel = 32, out_channel_1 = 128, out_channel_2=32,dilation=2**i))
        return nn.Sequential(*layers)

    def forward(self, x):
        # for test purpose only
        out0 = self.direct_conv(x)
        x1 = self.conv0(x)
        # local_pathway
        
        local_out = self.local_pathway(x1)
        # global pathway
        global_out = self.global_pathway(x1)
        # concatenation
        x2 = torch.cat((local_out, global_out),1) # need to check the concatenation dimension
        # conv final
        out1 = self.conv_final(x2)
        out = out0 + out1
            
        return out

# test
'''
my_DWAN = DWAN_network(img_channel=3)
print(sum(p.numel() for p in my_DWAN.parameters()))
'''