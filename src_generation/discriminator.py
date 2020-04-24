import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """Discriminator for the Cycle-GAN"""

    def __init__(self):
        """
        Creates Discriminator instance
        """
        super().__init__()
    
    def _build_conv_groups(self) -> list:
        """
        Builds convolutional 'group' consisting of torch.nn.Conv2d, (torch.nn.InstanceNorm2d), and torch.nn.LeakyReLU
        """
