import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, pool=False):
    """
    Creates a convolutional block with a convolutional layer, batch normalization, 
    and ReLU activation. Optionally adds a max pooling layer.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        pool (bool, optional): If True, adds a MaxPool2d layer. Default is False.
    Returns:
        nn.Sequential: A sequential container of the layers.
    """

    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    """
    A ResNet9 model implementation using PyTorch.
    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
    Methods:
        forward(xb):
            Defines the forward pass of the network.
            Args:
                xb (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after passing through the network.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out