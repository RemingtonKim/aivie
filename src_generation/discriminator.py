import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """Discriminator for the Cycle-GAN"""

    def __init__(self, start_in: int = 3, start_out: int = 64, kernel_size: int = 4, padding: int = 1, stride: int = 2, negative_slope: int = 0.2, num_groups: int = 4):
        """
        Creates Discriminator instance

        Args:
            start_in (int): number of channels in input image. Default is 3.
            start_out (int): number of channels produced by first torch.nn.Conv2d layer. Default is 64.
            kernel_size (int): height and width for the 2D convolutional window in torch.nn.Conv2d layer. Default is 4.
            padding (int): size of zero-padding in torch.nn.Conv2d layer. Default is 1.
            stride (int): stride argument for filter in torch.nn.Conv2d layer. Default is 2.
            negative_slope (int): determines the negative slope of the torch.nn.LeakyReLU layer. Default is 0.2
            num_groups (int): number of convolutional groups in model. Default is 4.
        """
        super().__init__()
        
        #creates a list to be passed into torch.nn.Sequential
        self._arg_model = self._build_conv_groups(in_channels=start_in, out_channels=start_out, kernel_size = kernel_size, padding = padding, stride = stride, negative_slope = negative_slope, normalization=False)
        
        self._in_channels = start_in
        self._out_channels = start_out

        #add groups with normalization to model
        for _ in range(num_groups-1):
            self._in_channels = self._out_channels
            self._out_channels *= 2
            self._arg_model += self._build_conv_groups(in_channels=self._in_channels, out_channels=self._out_channels, kernel_size = kernel_size, padding = padding, stride = stride, negative_slope = negative_slope)
        
        #add dense classification layer
        self._arg_model.append(nn.Conv2d(in_channels = self._out_channels, out_channels = 1, kernel_size = kernel_size, stride = stride, padding = padding))

        self.model = nn.Sequential(*self._arg_model)

    def _build_conv_groups(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, stride: int, negative_slope: int, normalization: bool  = True) -> list:
        """
        Builds convolutional 'group' consisting of torch.nn.Conv2d, (torch.nn.InstanceNorm2d), and torch.nn.LeakyReLU

        Args:
            in_channels (int): number of channels in input image
            out_channels (int): number of channels produced by torch.nn.Conv2d layer.
            kernel_size (int): height and width for the 2D convolutional window in torch.nn.Conv2d layer. 
            padding (int): size of zero-padding in torch.nn.Conv2d layer. 
            stride (int): stride argument for filter in torch.nn.Conv2d layer. 
            normalization (bool): determines whether or not the group will have normalization. Normalization if True, else no normalization.
            negative_slope (int): determines the negative slope of the torch.nn.LeakyReLU layer.

        Returns:
            list : list with a torch.nn.Conv2d, (torch.nn.InstanceNorm2d), and torch.nn.LeakyReLU to be a component of full Discriminator
        """
        cur = []
        
        #appends convolutional layer
        cur.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = stride, padding = padding))

        #appends normalization layer
        if normalization:
            cur.append(nn.InstanceNorm2d(num_features = out_channels))

        #appends LeakyReLU activation layer
        cur.append(nn.LeakyReLU(negative_slope = negative_slope, inplace = True))

        return cur


#check discriminator summary
if __name__ == '__main__':  
    print(Discriminator().model)