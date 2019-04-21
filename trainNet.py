import chess
import chess.pgn
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_Size = 2)
        self.conv2 = nn.Conv2d(64, 2, kernel_size = 3, padding = 1)
        self.lin1 = nn.Linear(2, 1)

    # def set_parameters(self):

    def forward(self, inputs):
        out = relu(self.conv1(inputs))
        out = self.pool(out)
        out = out.relu(self.conv2(out))
        out = self.lin1(out)

        return out