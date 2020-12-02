import torch
#import torchvision
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    ''' Encoder  - From pyramid bottom to op
    '''
    def __init__(self, in_channels, out_channels, sz=1):
        super(UpBlock, self).__init__()
        self.c1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                            stride=(sz, 2, 2), padding=1)
        
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = F.leaky_relu(x)
        return x
    
class DownBlock(nn.Module):
    ''' Encoder  - From pyramid bottom to op
    '''
    def __init__(self, in_channels, out_channels,sz=1):
        super(DownBlock, self).__init__()
        self.p1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3,
                                     stride=(sz, 2, 2), padding=1,
                                     output_padding=0)
        self.p2 = nn.Conv3d(2*out_channels, 2*out_channels, kernel_size=3, stride=1, padding=1)
        self.p3 = nn.Conv3d(2*out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, xin, c):
        x = self.p1(xin)
        xz = x.size()[2]
        cz = c.size()[2]
        pz = max(cz-xz, 0)
        x = F.pad(x, [0, 1, 0, 1, 0, pz])
        x = torch.cat([x, c], dim=1)
        x = self.p2(x)
        x = F.leaky_relu(x)
        x = self.p3(x)
        x = F.leaky_relu(x)
        return x
    
class unet3d(nn.Module):
    '''
    '''
    def __init__(self, params):
        super(unet3d, self).__init__()
        self.params = params
        self.create_layers()
    
    def create_layers(self):
        numf = 32
        nc = self.params['nchannels']
        self.up1 = UpBlock(nc, numf, sz=2)
        self.up2 = UpBlock(numf, numf, sz=1)
        self.up3 = UpBlock(numf, numf, sz=1)
        self.down2 = DownBlock(numf, numf, sz=1)
        self.down1 = DownBlock(numf, numf, sz=1)
        self.down0 = DownBlock(numf, nc, sz=2)
        self.out = nn.Conv3d(nc, 1, kernel_size=1)
        
    def forward(self, x):
        z1 = self.up1(x)
        z2 = self.up2(z1)
        z3 = self.up3(z2)
        z = self.down2(z3, z2)
        z = self.down1(z, z1)
        z = self.down0(z, x)
        z = self.out(z)
        return z
             

class DiceLoss(nn.Module):
    
    def __init__(self):
        super(DiceLoss, self).__init__()
        
    def forward(self, pred, target):
        pass