import numpy as np
import torch
import torch.nn as nn


def split_data(input_val):
    x = input_val[:, :-1]
    mask = input_val[:,-1].contiguous().view(x.shape[0],1,x.shape[2])
    return (x, mask)


class cust_Conv1d(nn.Module):
    """
    Simple custom Conv1d that passes forward mask information
    """
    def __init__(self, in_c=40, out_c=96, k_size=3, stride_size=1, pad_size=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=k_size, stride=stride_size, padding=pad_size)
        nn.init.xavier_normal(self.conv1.weight)
        self.leaky_ReLU = nn.LeakyReLU(0.01)


    def forward(self, input_val):
        (x, mask) = split_data(input_val)
        x = self.leaky_ReLU(self.conv1(x))
        #print(x.shape)
        #print(mask.shape)
        out_val = torch.cat((x, mask), dim=1)
        return out_val

class cust_Dropout(nn.Module):
    """
    Simple custom Conv1d that passes forward mask information
    """
    def __init__(self, prob=0.5, inplace_flag=False):
        super().__init__()
        self.drop = nn.Dropout(p=prob, inplace=inplace_flag)
        

    def forward(self, input_val):
        (x, mask) = split_data(input_val)
        x = self.drop(x)
        #x = self.leaky_ReLU(self.conv1(x))
        #print(x.shape)
        #print(mask.shape)
        out_val = torch.cat((x, mask), dim=1)
        return out_val

class cust_Pool(nn.Module):
    """
    Simple custom Pooling layer that transforms a CxF phoneme into a Cx1 output
    """
    def __init__(self,type):
        super().__init__()
        self.type = type

    def forward(self, input_val):
        (x, mask) = split_data(input_val)
        #print(x.shape)
        #print(mask.shape)
        x = x*mask
        #print(x.shape)
        return self.pooling_fn(x)
    
    def pooling_fn(self, x):
        if self.type == 'max':
            return torch.max(x, 2)
        elif self.type == 'mean':
            return torch.mean(x, 2)
        elif self.type == 'logsumexp':
            return torch.log(torch.sum(torch.exp(x), 2))

