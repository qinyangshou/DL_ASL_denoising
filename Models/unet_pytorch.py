import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """ for the convolutional block """
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)
    

class Unet_Down(nn.Module):
    
    """Downsampling block with maxpool and then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )
    def forward(self, x):
        return self.maxpool_conv(x)
    
class Unet_Up(nn.Module):
    
    """ Upsampling with nn.Upsample and double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size[3] - x1.size()[3]
        # may need to check with this part
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY //2, diffY - diffY//2])
        
        x = torch.cat([x2,x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)
    
class Unet(nn.Module):
    
    def __init__(self, n_channels, n_classes, bilinear = False):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.in_layer = DoubleConv(n_channels, 64)
        self.down1 = Unet_Down(64,128)
        self.down2 = Unet_Down(128,256)
        self.down3 = Unet_Down(256,512)
        factor = 2 if bilinear else 1
        self.down4 = Unet_Down(512, 1024//factor)
        self.up1 = Unet_Up(1024, 512//factor, bilinear)
        self.up2 = Unet_Up(512, 256//factor, bilinear)
        self.up3 = Unet_Up(256,128//factor, bilinear)
        self.up4 = Unet_Up(128,64, bilinear)
        self.out = OutConv(64, n_classes)
        
    def forword(self, x):
        
        x1 = self.in_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        out = self.out(x)
        
        return out

# test
my_unet = Unet(n_channels=1,n_classes=3)
print(sum(p.numel() for p in my_unet.parameters()))
    