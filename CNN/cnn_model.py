from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

# Reference
# Simonyan, K. and Zisserman, A., 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
VGG_CNF = {
    'A': [64, 'pool', 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'pool'],
    'B': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'pool'],
    'C': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool'],
    'D': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool'],
    'E': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 256, 'pool', 512, 512, 512, 512, 'pool', 512, 512, 512, 512, 'pool'],
}


class VGG(torch.nn.Module):
    def __init__(self, n_channels, layers, n_classes):
        """
        Initializes CNN object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          layers: list. vgg layers config
        """
        super(VGG, self).__init__()

        modules = []
        in_channels = n_channels
        for layer in layers:
            if layer != 'pool':
                modules.append(torch.nn.Conv2d(in_channels, layer, 3, padding=1))
                modules.append(torch.nn.BatchNorm2d(layer))
                modules.append(torch.nn.ReLU())
                in_channels = layer
            else:
                modules.append(torch.nn.MaxPool2d(3, stride=2, padding=1))

        self.conv = torch.nn.Sequential(*modules)
        # The input shape of CIFAR10 is 32x32x3, Thus the output shape of conv is 1x1x512.
        self.classfier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(512, n_classes),
            torch.nn.Softmax()
        )

    def forward(self, x):
        """
        Performs forward pass of the input.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        x = self.conv(x)
        x = x.view(-1, 512)
        out = self.classfier(x)
        return out
