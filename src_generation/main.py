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
import itertools
import numpy as np
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator

class CycleGAN:
    """complete Cycle-GAN class with two generators and two discriminators"""

    def __init__(self, lr: int = 0.0002, trainable: bool = False) -> None:
        """
        Creates Cycle-GAN instance
        
        Args:
            lr (int): learning rate for torch.optim.Adam in CycleGAN. Default is 0.0002.
            trainable (bool): determines whether or not the model can be trained. True for training, False for generating. Default is False
        """

        self.lr = lr
        self.trainable = trainable

        # set up torch device if GPU is available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # define generator and discriminator
        # generator and discriminator can be adjusted by overriding default params.
        self.generator_A2B = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)

        #define loss functions for generators and discriminators
        self.L1 = nn.L1Loss()
        self.MSE = nn.MSELoss()

        #optimizers for discriminators and generators
        self.optim_discriminator_A = optim.Adam(self.discriminator_A.parameters(), lr = self.lr, betas=(0.5, 0.999))
        self.optim_discriminator_B = optim.Adam(self.discriminator_B.parameters(), lr = self.lr, betas=(0.5, 0.999))
        self.optim_generator = optim.Adam(itertools.chain(self.generator_A2B.parameters(), self.generator_B2A.parameters()), lr = self.lr, betas=(0.5, 0.999))

        #data handling
        self.data_loader = None
    
    def load_data(self, path: str, batch_size: int = 32) -> None:
        """
        Loads data into the model

        Args:
            path (str): path to the directory containing the data
            batch_size (int): batch size for the data. Default is 32.
        """
        pass


    def train(self, epochs) -> None:
        """
        Trains Cycle-GAN model
        """
        #raise necessary errors
        if not self.trainable:
            raise RuntimeError('Cannot train model when trainable is set to False')
        if self.data_loader is None:
            raise RuntimeError('No data loaded into the model for training')

        for epoch in tqdm(range(epochs)):
            for _, batch in enumerate(self.data_loader):


   

    

    def generate(self):
        pass


#check Cycle-GAN model summary
if __name__ == '__main__':
    print(CycleGAN())