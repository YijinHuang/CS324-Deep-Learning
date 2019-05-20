from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class MLP(torch.nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()
        self.n_classes = n_classes

        layers = []
        d_in = n_inputs
        for d_out in n_hidden:
            layers.append(torch.nn.Linear(d_in, d_out))
            layers.append(torch.nn.ReLU())
            d_in = d_out

        layers.append(torch.nn.Linear(d_in, n_classes))
        self.parameters = torch.nn.ModuleList(layers)
        self.layers = layers

        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        for layer in self.layers:
            x = layer(x)
        out = self.softmax(x)
        return out


class CIFAR_MLP(torch.nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(CIFAR_MLP, self).__init__()
        self.n_classes = n_classes

        layers = []
        d_in = n_inputs
        for d_out in n_hidden:
            layers.append(torch.nn.Linear(d_in, d_out))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(d_out))
            d_in = d_out

        layers.append(torch.nn.Dropout(p=0.5))
        layers.append(torch.nn.Linear(d_in, n_classes))
        self.parameters = torch.nn.ModuleList(layers)
        self.layers = layers

        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        x = x.view(-1, 3*32*32)
        for layer in self.layers:
            x = layer(x)
        out = self.softmax(x)
        return out
