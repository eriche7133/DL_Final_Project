import numpy as np
import torch
import torch.utils.data
import collections
import torch.nn as nn
from torch.autograd import Variable
import collections

from custom_modules import *
from LSTM_State0 import LSTM_State0, LSTMCell_State0


def model_fn(args):
    model = FinalModule(args.input_dim, args.hidden_dim, args.speaker_num, args.encoder, args.decoder, args.pooling)
    return model


class SequencePooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        # Takes an input of size (L,B,D), converts it to size (L/2, B, D*2)
        (h, lengths) = nn.utils.rnn.pad_packed_sequence(input)
        # "Pooling" here
        if h.size()[0] % 2 == 1:
            h = h[:-1]
        h = torch.transpose(h,0,1)
        # print(h.size())
        h = h.contiguous().view(h.size()[0], h.size()[1]//2, h.size()[2]*2)
        h = torch.transpose(h,0,1)
        #print(type(np.array(lengths)//2))
        return nn.utils.rnn.pack_padded_sequence(h, np.array(lengths)//2)


class FinalModule(nn.Module):
    def __init__(self, input_dim = 40, hidden_dim = 256, speaker_num=1941, encoder='pBiLSTM', decoder='linear', pooling='max'):
        super().__init__()
        
        # self.cnn0 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
        # self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
        # self.cnn2 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
        
        # self.ReLU = nn.LeakyReLU(0.01)
        
        
        ## Encoder
        self.encoder = encoder
        self.decoder = decoder
        if self.encoder == 'pBiLSTM' or self.encoder == 'BiLSTM':
            self.batch_first = False
        else:
            self.batch_first = True
        if encoder == 'pBiLSTM':
            self.seq_pool = SequencePooling()
        
            self.rnns_encoder = nn.ModuleList([
                LSTM_State0(input_size=input_dim, hidden_size=hidden_dim,
                        batch_first=False, bidirectional=True),
                LSTM_State0(input_size=hidden_dim*4, hidden_size=hidden_dim,
                        batch_first=False, bidirectional=True),
                LSTM_State0(input_size=hidden_dim*4, hidden_size=hidden_dim,
                        batch_first=False,bidirectional=True)])
            self.final_rnn_encode = LSTM_State0(input_size=hidden_dim*4, hidden_size=hidden_dim,
                        batch_first=False,bidirectional=True)

            # self.keys = nn.Linear(in_features=hidden_dim*2,
                                  # out_features=key_dim)

            # self.values = nn.Linear(in_features=hidden_dim*2,
                                    # out_features=key_dim)
            # Output is (L,B,D) (length in time, batch, feature dimension) (D=hidden_dim*2)
        elif encoder == 'BiLSTM':
            self.rnns_encoder = nn.ModuleList([
                LSTM_State0(input_size=input_dim, hidden_size=hidden_dim,
                        batch_first=False, bidirectional=True),
                LSTM_State0(input_size=hidden_dim*2, hidden_size=hidden_dim,
                        batch_first=False, bidirectional=True),
                LSTM_State0(input_size=hidden_dim*2, hidden_size=hidden_dim,
                        batch_first=False,bidirectional=True)])
            
        elif encoder == 'CNN2D':
            self.conv1 = nn.Conv2d(40, 96, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
            self.conv6 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
            self.conv7 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
            self.conv8 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
            self.conv9 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
            self.conv10 = nn.Conv2d(192, 46, kernel_size=3, stride=1, padding=1)
            
            # self.conv_encoder = nn.ModuleList([
                    # nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=4, stride=2, padding=2),
                    # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)])
            self.maxpool = nn.MaxPool2d(4, stride=2, padding=2)
            print('Not Implemented')
        elif encoder == 'CNN1D':
            self.conv1 = nn.Conv1d(40, 96, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(96, 96, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv1d(96, 96, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv1d(96, 192, kernel_size=3, stride=1, padding=1)
            self.conv5 = nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1)
            self.conv6 = nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1)
            self.conv7 = nn.Conv1d(192, hidden_dim*2, kernel_size=3, stride=1, padding=1)
            self.conv8 = nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, stride=1, padding=1)
            self.conv9 = nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, stride=1, padding=1)
            self.conv10 = nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, stride=1, padding=1)
            
            self.leaky_ReLU = nn.LeakyReLU(0.2)
            self.maxpool = nn.MaxPool1d(4, stride=2, padding=2)
            
            # print('Not Implemented')
        elif encoder == 'HW2P2':
            self.conv1 = cust_Conv2d(in_c=1, out_c=96, k_size=3, stride_size=1, pad_size=1)
            self.conv2 = cust_Conv2d(in_c=96, out_c=96, k_size=3, stride_size=1, pad_size=1)
            self.conv3 = cust_Conv2d(in_c=96, out_c=96, k_size=3, stride_size=1, pad_size=1)
            self.conv4 = cust_Conv2d(in_c=96, out_c=192, k_size=3, stride_size=1, pad_size=1)
            self.conv5 = cust_Conv2d(in_c=192, out_c=192, k_size=3, stride_size=1, pad_size=1)
            self.conv6 = cust_Conv2d(in_c=192, out_c=speaker_num, k_size=3, stride_size=1, pad_size=1)
            self.conv7 = cust_Conv2d(in_c=speaker_num, out_c=speaker_num, k_size=3, stride_size=1, pad_size=1)
            self.conv8 = cust_Conv2d(in_c=speaker_num, out_c=speaker_num, k_size=3, stride_size=1, pad_size=1)
            self.conv9 = cust_Conv2d(in_c=speaker_num, out_c=speaker_num, k_size=3, stride_size=1, pad_size=1)
            self.conv10 = cust_Conv2d(in_c=speaker_num, out_c=speaker_num, k_size=3, stride_size=1, pad_size=1)
            self.pool1 = cust_Pool_nomask('mean')

        ## Decoder
        if decoder == 'linear':
            self.linear = nn.Linear(in_features=hidden_dim*2,
                                  out_features=speaker_num)
            self.pooling = cust_Pool(pooling, self.batch_first)
        elif decoder == 'MLP':
            self.mlp = nn.ModuleList([
                nn.Linear(in_features = hidden_dim*2, out_features = hidden_dim*2),
                nn.Linear(in_features = hidden_dim*2, out_features = hidden_dim*2),
                nn.Linear(in_features = hidden_dim*2, out_features = hidden_dim)])
            self.mlp_fin = nn.Linear(in_features = hidden_dim, out_features = speaker_num)
            self.pooling = cust_Pool(pooling, self.batch_first)
            self.activation = nn.LeakyReLU(0.2)
        elif decoder == 'FC':
            # Since input is variable length, can this be done?
            # self.resizer = 
            print('Not Implemented')
        elif decoder == 'CNN2D':
            self.conv_decoder = nn.ModuleList([
                    nn.Conv2d(in_channels=1, out_channels=hidden_dim*2, kernel_size=4, stride=2, padding=2),
                    nn.Conv2d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=4, stride=2, padding=2),
                    nn.Conv2d(in_channels=hidden_dim*2, out_channels=hidden_dim, kernel_size=4, stride=2, padding=2)])
            self.relu = nn.ReLU()
            self.linear = nn.Linear(in_features=hidden_dim, out_features=speaker_num)
            if pooling == 'max':
                self.pool = nn.MaxPool2d(4, stride=2)
            elif pooling == 'mean':
                self.pool = nn.AvgPool2d(4, stride=2)
            self.pooling = cust_Pool(pooling, self.batch_first)
            print('Not Implemented')
        elif decoder == 'CNN1D':
            self.conv_decoder = nn.ModuleList([
                    nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=4, stride=2, padding=2),
                    nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=4, stride=2, padding=2),
                    nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim, kernel_size=4, stride=2, padding=2)])
            self.relu = nn.ReLU()
            self.linear = nn.Linear(in_features=hidden_dim, out_features=speaker_num)
            if pooling == 'max':
                self.pool = nn.MaxPool1d(4, stride=2)
            elif pooling == 'mean':
                self.pool = nn.AvgPool1d(4, stride=2)
            self.pooling = cust_Pool(pooling, self.batch_first)
            print('Not Implemented')
        elif decoder == 'attention':
            self.attention = nn.Linear(in_features=hidden_dim*2,
                                       out_features=speaker_num)
            self.weight_W = nn.Linear(in_features=hidden_dim*2,
                                      out_features=1)
            self.softmax2 = nn.Softmax(dim=2)
            self.softmax1 = nn.Softmax(dim=1)


    def forward(self, *input):
        h = input[0]
        utt_lengths = input[1]
        print_values = False
        if print_values:
            print('h size: {}'.format(h.size()))
            print('utt_lengths size: {}'.format(utt_lengths.size()))
            print('Before Encoder')
        #### Encoder
        ##### pack here
        if self.encoder == 'pBiLSTM':
            batch_size = h.size()[1]
            h = nn.utils.rnn.pack_padded_sequence(h, utt_lengths.cpu().data.int().numpy())
            states = [None] * len(self.rnns_encoder)
            for j, rnn in enumerate(self.rnns_encoder):
                rnn.flatten_parameters()
                h, state = rnn(h, states[j], batch_size)
                # print(type(state))
                #print(h.size())
                h = self.seq_pool(h)
                utt_lengths /= 2
                states[j] = state
                #print(h.size())
                #states.append(state)
            
            h, state = self.final_rnn_encode(h, None, batch_size)
            states.append(state)
            # Output is of size (L,B,D)
            
            ###### unpack here
            (h, _) = nn.utils.rnn.pad_packed_sequence(h)
        elif self.encoder == 'BiLSTM':
            batch_size = h.size()[1]
            h = nn.utils.rnn.pack_padded_sequence(h, utt_lengths.cpu().data.int().numpy())
            states = [None] * len(self.rnns_encoder)
            for j, rnn in enumerate(self.rnns_encoder):
                rnn.flatten_parameters()
                h, state = rnn(h, states[j], batch_size)
                # print(type(state))
                #print(h.size())
                #print(h.size())
                # states.append(state)
                states[j] = state
            (h, _) = nn.utils.rnn.pad_packed_sequence(h)
        elif self.encoder == 'CNN2D':
            # Switch to batch first
            h = torch.transpose(h,0,1)
            
        elif self.encoder == 'CNN1D':
            # print(h.size())
            # Switch from (L,B,C) to (B,L,C)
            h = torch.transpose(h,0,1)
            # print(h.size())
            # Switch from (B,L,C) to (B,C,L)
            h = torch.transpose(h,1,2)
            # print(h.size())
            
            h = self.conv1(h)
            h = self.leaky_ReLU(h)
            h = self.conv2(h)
            h = self.leaky_ReLU(h)
            h = self.maxpool(h)
            # print(h.size())
            utt_lengths = utt_lengths/2+1
            # print(h.size())
            
            h = self.conv3(h)
            h = self.leaky_ReLU(h)
            h = self.conv4(h)
            h = self.leaky_ReLU(h)
            h = self.maxpool(h)
            # print(h.size())
            utt_lengths = utt_lengths/2+1
            # print(h.size())
            
            h = self.conv5(h)
            h = self.leaky_ReLU(h)
            h = self.conv6(h)
            h = self.leaky_ReLU(h)
            h = self.maxpool(h)
            # print(h.size())
            utt_lengths = utt_lengths/2+1
            # print(h.size())
            
            h = self.conv7(h)
            h = self.leaky_ReLU(h)
            h = self.conv8(h)
            h = self.leaky_ReLU(h)
            h = self.maxpool(h)
            # print(h.size())
            utt_lengths = utt_lengths/2+1
            # print(h.size())
            
            h = self.conv9(h)
            h = self.leaky_ReLU(h)
            h = self.conv10(h)
            h = self.leaky_ReLU(h)
            
            # Switch from (B,C,L) to (B,L,C)
            # print(h.size())
            h = torch.transpose(h,1,2)
            # print(h.size())
            
            # print('Not Implemented')
        elif self.encoder == 'HW2P2':
            # Switch from (L,B,C) to (B,L,C)
            h = torch.transpose(h,0,1)
            # Turn (B,L,C) data into (B,1,L,C) data
            h = torch.unsqueeze(h,1)
            # print(h.size())
            h = self.conv1(h)
            h = self.conv2(h)
            h = self.conv3(h)
            h = self.conv4(h)
            h = self.conv5(h)
            h = self.conv6(h)
            h = self.conv7(h)
            h = self.conv8(h)
            h = self.conv9(h)
            h = self.conv10(h)
            h = self.pool1(h)

            # print(h.size())
        
        # h = self.projection(h) + 1
        
        ## Decoder
        if self.decoder == 'linear':
            ## Currently assumes (L,B,D)
            if print_values:
                print('Decoder')
                print('h size: {}'.format(h.size()))
            h = self.linear(h)
            if print_values:
                print('h size: {}'.format(h.size()))
            mask = create_mask(utt_lengths, self.batch_first)
            h = self.pooling(h, mask)
            if print_values:
                print('h size: {}'.format(h.size()))
        elif self.decoder == 'MLP':
            for linear_layer in self.mlp:
                h = linear_layer(h)
                h = self.activation(h)
            h = self.mlp_fin(h)
            # print(utt_lengths[0])
            # print(h.size()[1])
            assert int(utt_lengths[0].data) == h.size()[1]
            mask = create_mask(utt_lengths, self.batch_first)
            # print(mask.size())
            # if self.batch_first:
                # print(torch.sum(mask,1))
            # else:
                # print(torch.sum(mask,0))
            # print(utt_lengths.data)
            # print(mask)
            h = self.pooling(h, mask, utt_lengths)
        elif self.decoder == 'FC':
            print('Not Implemented')
        elif self.decoder == 'CNN2D':
            # transpose from (L,B,D) to (B,L,D)
            h = torch.transpose(h,0,1)
            # reshape from (B,L,D) to (B,1,L,D)
            h = torch.contiguous().view(h.size()[0],1,h.size()[1],h.size()[2])
            for cnn in self.conv_decoder:
                h = cnn(h) # what does this do the the lengths?
                h = self.relu(h)
                h = self.pool(h) # what does this do to the lengths?
            h = self.linear(h)
            mask = create_mask(utt_lengths, self.batch_first)
            h = self.pool(h, mask)
            print('Not Implemented')
        elif self.decoder == 'CNN1D':
            # transpose from (L,B,D) to (B,L,D)
            h = torch.transpose(h,0,1)
            # transpose from (B,L,D) to (B,D,L)
            h = torch.transpose(h,1,2)
            for cnn in self.conv_decoder:
                h = cnn(h) # what does this do the the lengths?
                h = self.relu(h)
                h = self.pool(h) # what does this do to the lengths?
            h = self.linear(h)
            mask = create_mask(utt_lengths, self.batch_first)
            h = self.pool(h, mask)
            print('Not Implemented')
        elif self.decoder == 'attention':
            ## Still need masks somewhere!!!
            # transpose from (L,B,D) to (B,L,D)
            mask = create_mask(utt_lengths, self.batch_first)
            h = torch.transpose(h,0,1)
            mask = torch.transpose(mask,0,1)
            # u should be (B,L,K)
            u = self.attention(h)
            # w should be (B,L,K) still
            w = self.softmax2(u)
            w = w*mask
            w = w/torch.sum(w,2,keepdim=True)
            # h is (B,L,D), w is (B,L,K), we want 
            # transpose u from (B,L,K) to (B,K,L)
            w = torch.transpose(w,1,2)
            # batch matrix multiply (B,K,L) x (B,L,D) into (B,K,D)
            f = torch.bmm(w,h)
            # transform f into fsm (B,K,D) into (B,K,1)
            fsm = self.weight_W(f)
            # view from (B,K,1) to (B,K)
            fsm = fsm.contiguous().view(fsm.size()[0],fsm.size()[1])
            output = self.softmax1(fsm)
            h = output


        return h
        # return (final_out, utt_lengths, label_lengths)


def create_mask(lengths, batch_first=False):
    #####
    ## batch_first, etc.
    # mask = torch.autograd.Variable(torch.arange(lengths[0],out=lengths.data.new()).unsqueeze(1), requires_grad=False) < lengths
    # if batch_first:
        # mask = torch.transpose(mask,0,1)
    
    if batch_first:
        mask = torch.zeros(lengths.size()[0], int(lengths[0]), 1)
        for i in range(lengths.size()[0]):
            mask[i,0:int(lengths[i]), 0] = 1
    else:
        mask = torch.zeros(int(lengths[0]), lengths.size()[0], 1)
        for i in range(lengths.size()[0]):
            mask[0:int(lengths[i]), i, 0] = 1
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        mask = mask.cuda()
    mask = torch.autograd.Variable(mask, requires_grad=False)
    return mask


class cust_Pool(nn.Module):
    """
    Simple custom Pooling layer that transforms a NxCxF phoneme into a Nx1xF output
    """
    def __init__(self,type,batch_first=False):
        super().__init__()
        self.type = type
        self.batch_first = batch_first

    def forward(self, x, mask, lengths):
        #(x, mask) = split_data(input_val)
        #print(x.shape)
        #print(mask.shape)
        # print(x.size())
        # print(mask.size())
        x = x*mask
        #print(x.shape)
        return self.pooling_fn(x,lengths)
    
    def pooling_fn(self, x, lengths):
        if self.batch_first:
            pool_dim = 1
        else:
            pool_dim = 0
        if self.type == 'max':
            h = torch.max(x, pool_dim)[0]
            return h
        elif self.type == 'mean':
            sum_val = torch.sum(x, pool_dim)
            # print(sum_val.size())
            # print(lengths.size())
            return sum_val/lengths.float().unsqueeze(1)
        elif self.type == 'logsumexp':
            return torch.log(torch.sum(torch.exp(x), pool_dim))


class cust_Pool_nomask(nn.Module):
    """
    Simple custom Pooling layer that transforms a NxCxF phoneme into a Nx1xF output
    """
    def __init__(self,type,batch_first=False):
        super().__init__()
        self.type = type
        self.batch_first = batch_first

    def forward(self, x):
        #(x, mask) = split_data(input_val)
        #print(x.shape)
        #print(mask.shape)
        # print(x.size())
        # print(mask.size())
        #print(x.shape)
        return self.pooling_fn(x)
    
    def pooling_fn(self, x):
        if self.batch_first:
            pool_dim = 1
        else:
            pool_dim = 0
        if self.type == 'max':
            h = torch.max(x, pool_dim)[0]
            return h
        elif self.type == 'mean':
            l = x.size()[3]
            w = x.size()[2]
            sum_val = torch.sum(torch.sum(x, 3),2)
            # print(sum_val.size())
            # print(lengths.size())
            return sum_val/l/w
        elif self.type == 'logsumexp':
            return torch.log(torch.sum(torch.exp(x), pool_dim))


