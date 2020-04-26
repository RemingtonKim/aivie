"""
A pytorch implementation of the Cycle-GAN architecture used for generating art in aivie.
The original paper by Zhu et al. can be found at: https://arxiv.org/pdf/1703.10593.pdf
Their more complete and performant official pytorch implementation can be found at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from generator import Generator
from discriminator import Discriminator

class CycleGAN:
    """complete Cycle-GAN class with two generators and two discriminators"""

    def __init__(self, lr: int = 0.0002, weight_init_type: str = 'normal') -> None:
        """
        Creates Cycle-GAN instance
        
        Args:
            lr (int): learning rate for torch.optim.Adam in CycleGAN. Default is 0.0002.
            weight_init_type (str): weight initialization type for layers. Options are 'normal', 'kaiming', and 'xavier'. Default is 'normal'
        """

        self.lr = lr
        self.weight_init_type = weight_init_type

        # set up torch device if GPU is available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # define generator and discriminator
        # generator and discriminator can be adjusted by overriding default params.
        self.generator_A2B = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)

        #define loss functions for generators and discriminators
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_gan = nn.MSELoss()

        #

        

    def train(self):
        pass

    def generate(self):
        pass


#check Cycle-GAN model summary
if __name__ == '__main__':
    print(CycleGAN())