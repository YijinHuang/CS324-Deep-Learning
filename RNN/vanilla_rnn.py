from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embed = nn.Embedding(10, input_dim)
        self.hx = nn.Linear(input_dim, hidden_dim)
        self.hh = nn.Linear(hidden_dim, hidden_dim)
        self.oh = nn.Linear(hidden_dim, output_dim)
        self.y = nn.Softmax()

    def forward(self, x):
        x = self.embed(x)
        hs = []
        for i in range(self.seq_length):
            if i == 0:
                h = torch.tanh(self.hx(x[:, i, :]))
            else:
                h = torch.tanh(self.hx(x[:, i, :]) + self.hh(hs[i-1]))
            hs.append(h)

        o = self.oh(h)
        out = self.y(o)

        return out