import torch
import torchvision
import torch.nn as nn
import torch.autograd as autograd
from collections import OrderedDict

class ResNetBlock(nn.Module):
    """
    A residual block used in the generator of this Cycle-GAN.
    """
    def __init__(self, in_channels: int, stride: int = 2, padding: int = 1, kernel_size : int = 4):
        """
        Creates a ResNetBlock instance
        
        Args:
            in_channels (int) : number of channels in input image
            stride (int) : stride argument for filter in torch.nn.Conv2d layer. Default is 2.
            padding (int) : size of padding for left, right, top and bottom for torch.nn.ReflectionPad2d. Default is 1.
            kernel_size (int) : height and width for the 2D convolutional window in torch.nn.Conv2d layer. Default is 4.
        """
        super().__init__()

        #set in_channels and out_channels to be the same for convolutional layers
        self._out_channels = in_channels 
        
        #build model
        self.model = nn.Sequential(
            nn.ReflectionPad2d(padding = padding),
            nn.Conv2d(in_channels = in_channels, out_channels = self._out_channels, kernel_size=kernel_size, stride=stride),
            nn.InstanceNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding = padding),
            nn.Conv2d(in_channels = in_channels, out_channels = self._out_channels, kernel_size=kernel_size, stride=stride),
            nn.InstanceNorm2d(num_features=in_channels)
        )
    
    def forward(self, x):
        #concatenates the tensors
        return x + self.model(x)


class Generator(nn.Module):
    """Generator for the Cycle-GAN"""

    def __init__(self):
        """
        Creates a Generator instance
        """
        super().__init__()


