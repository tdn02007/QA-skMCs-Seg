import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.BatchNorm2d(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResidualGenerator(nn.Module):
    def __init__(self, in_channels=2, dim=64, n_downsample=2,  n_upsample=2):
        super(ResidualGenerator, self).__init__()

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        dim = 64

        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(5):
            layers += [ResidualBlock(dim)]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, dim, 7)]

        self.model_blocks = nn.Sequential(*layers)

        cell_layers = [nn.ReflectionPad2d(2), nn.Conv2d(dim, 1, 5)]
        self.cell_blocks = nn.Sequential(*cell_layers)

        nucl_layers = [nn.ReflectionPad2d(2), nn.Conv2d(dim, 1, 5)]
        self.nucl_blocks = nn.Sequential(*nucl_layers)

        mask_layers = [nn.ReflectionPad2d(2), nn.Conv2d(dim, 1, 5)]
        self.mask_blocks = nn.Sequential(*mask_layers)

    def forward(self, x):
        x = self.model_blocks(x)
        cell = self.cell_blocks(x).reshape(x.shape[0], x.shape[2], x.shape[3])
        nucl = self.nucl_blocks(x).reshape(x.shape[0], x.shape[2], x.shape[3])
        mask = self.mask_blocks(x).reshape(x.shape[0], x.shape[2], x.shape[3])
        return cell, nucl, mask


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice