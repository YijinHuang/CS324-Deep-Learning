from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTM, self).__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        self.gx = nn.Linear(input_dim, hidden_dim)
        self.gh = nn.Linear(hidden_dim, hidden_dim)

        self.ix = nn.Linear(input_dim, hidden_dim)
        self.ih = nn.Linear(hidden_dim, hidden_dim)

        self.fx = nn.Linear(input_dim, hidden_dim)
        self.fh = nn.Linear(hidden_dim, hidden_dim)

        self.ox = nn.Linear(input_dim, hidden_dim)
        self.oh = nn.Linear(hidden_dim, hidden_dim)

        self.ph = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(2)
        h = torch.zeros((self.hidden_dim, self.hidden_dim), dtype=torch.float32).cuda()
        c = torch.zeros((self.hidden_dim, self.hidden_dim), dtype=torch.float32).cuda()

        for idx in range(self.seq_length):
            g = torch.tanh(self.gx(x[:, idx]) + self.gh(h))
            i = torch.sigmoid(self.ix(x[:, idx]) + self.ih(h))
            f = torch.sigmoid(self.fx(x[:, idx]) + self.fh(h))
            o = torch.sigmoid(self.ox(x[:, idx]) + self.oh(h))

            c = g * i + c * f
            h = torch.tanh(c) * o

        out = F.softmax(self.ph(h), dim=1)
        return out
