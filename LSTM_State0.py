import numpy as np
import torch
import torch.nn as nn


class LSTM_State0(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, num_layers=1,
                 bias=True, batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                      bias=bias, batch_first=batch_first, dropout=dropout,
                                      bidirectional=bidirectional)
        self.hidden_size = hidden_size
        if bidirectional:
            self.bidim = 2
        else:
            self.bidim = 1
        self.h0 = nn.Parameter(torch.FloatTensor(self.bidim, 1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(self.bidim, 1, self.hidden_size).zero_())


    def forward(self, input, state, batch_size):
        # input = nn.utils.rnn.pack_padded_sequence(input, lengths)
        # if self.LSTM.batch_first:
            # batch_size = input.size()[0]
        # else:
            # batch_size = input.size()[1]
        if state is None:
            h = self.h0.expand(-1, batch_size, -1)
            c = self.c0.expand(-1, batch_size, -1)
        else:
            (h, c) = state
        # if h is None:
            # h = self.h0.expand(-1, batch_size, -1)
        # if c is None:
            # c = self.c0.expand(-1, batch_size, -1)
        h = h.contiguous()
        c = c.contiguous()
        self.LSTM.flatten_parameters()
        return self.LSTM(input,(h,c))
    
    
    def flatten_parameters(self):
        self.LSTM.flatten_parameters()


class LSTMCell_State0(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, bias=True):
        super(LSTMCell_State0, self).__init__()
        self.LSTMCell = nn.LSTMCell(input_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.h0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_().contiguous())
        self.c0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_().contiguous())


    def forward(self, input, state):
        if state is None:
            batch_size = input.size()[0]
            h = self.h0.expand(batch_size, -1)
            c = self.c0.expand(batch_size, -1)
        else:
            (h, c) = state
        # if h is None:
            # h = self.h0.expand(batch_size, -1)
        # if c is None:
            # c = self.c0.expand(batch_size, -1)
        print_values = False
        if print_values:
            print('LSTMCell')
            print('h.size() = {}'.format(h.size()))
            print('c.size() = {}'.format(c.size()))
        return self.LSTMCell(input, (h,c))


