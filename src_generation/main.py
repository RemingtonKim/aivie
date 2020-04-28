"""
A pytorch implementation of the Cycle-GAN architecture used for generating art in aivie.
The original paper by Zhu et al. can be found at: https://arxiv.org/pdf/1703.10593.pdf
This implementation is based on their more complete and performant official pytorch implementation, which can be found at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
This implementation was inspired by https://github.com/aitorzip/PyTorch-CycleGAN. However, this implementation is not a verbatim copy of the aforementioned implementation, as it was heavily modified and completely rewritten for practical and educational purposes. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torch.autograd as autograd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import itertools
import numpy as np
import os
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
from data_dataset import DatasetAB


class CycleGAN:
    """complete Cycle-GAN class with two generators and two discriminators"""

    def __init__(self, lr: int = 0.0002, trainable: bool = False, lambda_a: float = 10.0, lambda_b: float = 10.0, lambda_identity: float = 0.5) -> None:
        """
        Creates Cycle-GAN instance
        
        Args:
            lr (int): learning rate for torch.optim.Adam in CycleGAN. Default is 0.0002.
            trainable (bool): determines whether or not the model can be trained. True for training, False for generating. Default is False
            lambda_a (float):
            lambda_b (float):
            lambda_identity (float): 
        """

        self.lr = lr
        self.trainable = trainable
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.lambda_identity = lambda_identity

        # set up torch device if GPU is available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # define generator and discriminator
        # generator and discriminator can be adjusted by overriding default params.
        self.generator_A2B = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)

        #define loss functions for generators and discriminators
        self.criterion_idt = nn.L1Loss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_gan = nn.MSELoss()

        #optimizers for discriminators and generators
        self.optim_discriminator_A = optim.Adam(self.discriminator_A.parameters(), lr = self.lr, betas=(0.5, 0.999))
        self.optim_discriminator_B = optim.Adam(self.discriminator_B.parameters(), lr = self.lr, betas=(0.5, 0.999))
        self.optim_generator = optim.Adam(itertools.chain(self.generator_A2B.parameters(), self.generator_B2A.parameters()), lr = self.lr, betas=(0.5, 0.999))

        #data handling
        self.data_loader = None
    
    def load_data(self, path_A: str, path_B: str, batch_size: int = 32) -> None:
        """
        Loads data into the model

        Args:
            path_A (str): path to the directory containing the data for A
            path_B (str): path to the directory containing the data for B
            batch_size (int): batch size for the data. Default is 32.
        """

        #transformations to be applied to data. Modify as needed
        transformations = transforms.Compose(transforms=[
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])

        #apply transformations to image in given directory.
        images = DatasetAB(
            datasets.ImageFolder(path_A, transform=transformations),
            datasets.ImageFolder(path_B, transform=transformations)       
        )
        #load dataset into dataloader for use in training
        self.data_loader = data.DataLoader(
            dataset=images,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

        #ground truth for fake and real images
        self.target_reals = torch.Tensor(np.ones(batch_size)).to(self.device)
        self.target_fakes = torch.Tensor(np.zeros(batch_size)).to(self.device)

    def train(self, epochs: int = 50) -> None:
        """
        Trains Cycle-GAN model

        Args:
            epochs (int): number of epochs to train Cycle-GAN model for. Default is 50.
        """
        #raise necessary errors
        if not self.trainable:
            raise RuntimeError('Cannot train model when trainable is set to False')
        if self.data_loader is None:
            raise RuntimeError('No data loaded into the model for training')
        
        # iterate through epochs
        for epoch in range(epochs):
            #iterate through batches
            for _, batch in tqdm(enumerate(self.data_loader)):

                #current batch             
                batch_real_a = autograd.Variable(batch[0]).to(self.device)
                batch_real_b = autograd.Variable(batch[1]).to(self.device)
                print(batch_real_a.shape)
                print(batch_real_b.shape)
                
                
                #generator training
                self.optim_generator.zero_grad()

                #GAN losses
                fake_b = self.generator_A2B(batch_real_a)
                pred_fake = self.discriminator_B(fake_b)
                generator_A2B_loss = self.criterion_gan(pred_fake, fake_b)

                fake_a = self.generator_B2A(batch_real_b)
                pred_fake = self.discriminator_A(fake_a)
                generator_B2A_loss = self.criterion_gan(pred_fake, fake_a)

                #forward and backward cycle losses
                reconstructed_a = self.generator_B2A(fake_b)
                cycle_ABA_loss = self.criterion_cycle(reconstructed_a, batch_real_a) * self.lambda_a

                reconstructed_b = self.generator_A2B(fake_a)
                cycle_BAB_loss = self.criterion_cycle(reconstructed_b, batch_real_b) * self.lambda_b

                #identity losses
                identity_a = self.generator_B2A(batch_real_a)
                identity_a_loss = self.criterion_idt(identity_a, batch_real_a) * self.lambda_a * self.lambda_identity

                identity_b = self.generator_A2B(batch_real_b)
                identity_b_loss = self.criterion_idt(identity_b, batch_real_b) * self.lambda_b * self.lambda_identity

                generator_losses = generator_A2B_loss + generator_B2A_loss + cycle_ABA_loss + cycle_BAB_loss + identity_a_loss + identity_b_loss
                generator_losses.backward()

                self.optim_generator.step()

                #discriminator training
                self.optim_discriminator_A.zero_grad()

                pred_real = self.discriminator_A(batch_real_a)
                loss_d_real = self.criterion_gan(pred_real, self.target_reals)

                pred_fake = self.discriminator_A(fake_a.detach())
                loss_d_fake = self.criterion_gan(pred_fake, self.target_fakes)

                loss_discriminator_a = 0.5 * (loss_d_real + loss_d_fake)
                loss_discriminator_a.backward()

                self.optim_discriminator_A.step()

                self.optim_discriminator_B.zero_grad()

                pred_real = self.discriminator_A(batch_real_a)
                loss_d_real = self.criterion_gan(pred_real, self.target_reals)

                pred_fake = self.discriminator_B(fake_b.detach())
                loss_d_fake = self.criterion_gan(pred_fake, self.target_fakes)

                loss_discriminator_b = 0.5 * (loss_d_real + loss_d_fake)
                loss_discriminator_b.backward()

                self.optim_discriminator_B.step()
                
    def generate(self):
        pass


#check Cycle-GAN model summary
if __name__ == '__main__':
    c = CycleGAN()
    c.trainable = True
    c.load_data(path_A='../data_img_A', path_B='../data_img_B', batch_size=8)
    c.train(epochs=1)
    print('done')