import torch
import torch.nn as nn
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = None):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        if self.stride is not None:
            self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, X):
        Y = X
        Y = self.conv(Y)
        if self.stride is not None:
            Y = self.pool(Y)
        Y = self.relu(Y)

        return Y


class ActionPredictionModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.conv1 = ConvBlock(in_channels=args.sequence_length, out_channels=64, kernel_size=7, stride=5)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=5, stride=3)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2)

        self.flatten = nn.Flatten()

        self.readout = nn.Linear(10240, args.num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        Y = X

        Y = self.conv1(Y)
        Y = self.conv2(Y)
        Y = self.conv3(Y)
        Y = self.conv4(Y)
        Y = self.flatten(Y)
        Y = self.readout(Y)

        Y = self.softmax(Y)

        return Y