import torch
import torchvision
import torch.nn as nn
import torch.autograd as autograd

class ResNetBlock(nn.Module):
    """
    A residual block used in the generator of this Cycle-GAN.
    """
    def __init__(self, in_channels: int, r_padding: int = 1, kernel_size : int = 3):
        """
        Creates a ResNetBlock instance
        
        Args:
            in_channels (int) : number of channels in input image
            r_padding (int) : size of padding for left, right, top and bottom for torch.nn.ReflectionPad2d. Default is 1.
            kernel_size (int) : height and width for the 2D convolutional window in torch.nn.Conv2d layer. Default is 3.
        """
        super().__init__()

        #set in_channels and out_channels to be the same for convolutional layers
        self._out_channels = in_channels 
        
        #build model
        self.model = nn.Sequential(
            nn.ReflectionPad2d(padding = r_padding),
            nn.Conv2d(in_channels = in_channels, out_channels = self._out_channels, kernel_size=kernel_size),
            nn.InstanceNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding = r_padding),
            nn.Conv2d(in_channels = in_channels, out_channels = self._out_channels, kernel_size=kernel_size),
            nn.InstanceNorm2d(num_features=in_channels)
        )
    
    def forward(self, x):
        """Concatenates tensors in forward"""
        return x + self.model(x)


class Generator(nn.Module):
    """Generator for the Cycle-GAN"""

    def __init__(self, start_in: int = 3, start_out: int = 64, end_out: int = 3, ends_kernel_size: int = 7, mid_kernel_size: int = 3, r_padding: int = 3, padding: int = 1, stride: int = 2, n_resnet = 9) -> None:
        """
        Creates a Generator instance. Will have c7s1-64, d128, d256, R256, R256, R256, R256, R256,R256, R256, R256, R256, u256, u128, c7s1-3 architecture. 

        Args:
            start_in (int): number of channels in input tensor. Default is 3.
            start_out (int): number of channels produced by first torch.nn.Conv2d layer. Default is 64.
            end_out (int): number of channels in final tensor. Default is 3.
            ends_kernel_size (int): height and width for the 2D convolutional window in first and last torch.nn.Conv2d layer. Default is 7.
            mid_kernel_size (int): height and width for the 2D convolutional window in middle torch.nn.Conv2d layers. Default is 3.
            r_padding (int) : size of padding for left, right, top and bottom for torch.nn.ReflectionPad2d. Default is 3.
            padding (int): size of zero-padding in torch.nn.Conv2d layer. Default is 1.
            stride (int) : stride argument for filter in torch.nn.Conv2d layer. Default is 2.
            n_resnet (int) : determines the number of resnet blocks in model. Default is 9.
        """
        super().__init__()
        
        # define constants
        self.NUM_DOWNSAMPLE = 2
        self.NUM_UPSAMPLE = 2

        # c7s1-64 block
        self._arg_model = [
            nn.ReflectionPad2d(padding=r_padding),
            nn.Conv2d(in_channels = start_in, out_channels = start_out, kernel_size = ends_kernel_size),
            nn.InstanceNorm2d(num_features=start_out),
            nn.ReLU(inplace=True)
        ]

        self._in_channels = start_in
        self._out_channels = start_out

        #d128 & d256 block
        for _ in range(self.NUM_DOWNSAMPLE):
            self._in_channels = self._out_channels
            self._out_channels *= 2
            self._arg_model += self._downsample(in_channels=self._in_channels, out_channels=self._out_channels, padding = padding, kernel_size=mid_kernel_size, stride=stride)

        #R256 blocks
        for _ in range(n_resnet):
            self._arg_model.append(ResNetBlock(in_channels=self._out_channels))
        
        
        #u128 & u64 blocks
        for _ in range(self.NUM_UPSAMPLE):
            self._in_channels = self._out_channels
            self._out_channels = self._in_channels // 2
            self._arg_model += self._upsample(in_channels=self._in_channels, out_channels=self._out_channels, kernel_size=mid_kernel_size, padding=padding, output_padding=padding, stride = stride)

        #output layer
        self._arg_model += [
            nn.ReflectionPad2d(padding=r_padding),
            nn.Conv2d(in_channels = self._out_channels, out_channels=end_out, kernel_size=ends_kernel_size),
            nn.Tanh()
        ]

        #build model
        self.model = nn.Sequential(*self._arg_model)


    def forward(self, x):
        """Standard forward"""
        return self.model(x)

    
    def _downsample(self, in_channels: int, out_channels: int, padding: int, kernel_size: int, stride: int) -> list:
        """
        Creates downsampling block for generator

        Args:
            in_channels (int): number of channels in input tensor
            out_channels (int): number of channels produced by torch.nn.Conv2d layer
            padding (int): size of zero-padding in torch.nn.Conv2d layer.
            kernel_size (int): height and width for the 2D convolutional window in torch.nn.Conv2d layer.
            stride (int) : stride argument for filter in torch.nn.Conv2d layer.
        Returns:
            list: list with torch.nn.Conv2d, torch.nn.InstanceNorm2d, nn.ReLU
        """
        cur = [
            nn.Conv2d(in_channels= in_channels, out_channels=out_channels, kernel_size=kernel_size, padding = padding, stride=stride),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        ]
        return cur


    def _upsample(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, output_padding: int, stride: int) -> list:
        """
        Creates upsampling block for generator

        Args:
            in_channels (int): number of channels in input tensor
            out_channels (int): number of channels produced by torch.nn.Conv2d layer
            padding (int): size of zero-padding in torch.nn.Conv2d layer.
            output_padding (int) : controls additional size added to output of nn.ConvTranspose2d
            stride (int) : stride argument for filter in torch.nn.Conv2d layer.
            kernel_size (int): height and width for the 2D convolutional window in torch.nn.ConvTranspose2d layer.
        Returns:
            list: list with torch.nn.ConvTranspose2d, torch.nn.InstanceNorm2d, nn.ReLU
        """
        cur = [
            nn.ConvTranspose2d(in_channels= in_channels, out_channels=out_channels, kernel_size=kernel_size, padding = padding, output_padding=output_padding, stride=stride),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        ]
        return cur

#check model summary
if __name__ == '__main__':
    from torchsummary import summary
    summary(Generator(), (3, 256, 256), device='cpu')