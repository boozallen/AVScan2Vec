import sys

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .globalvars import *


class Highway(nn.Module):

    def __init__(self, in_D):
        """Highway class adapted from https://github.com/helboukkouri/character-bert/blob/main/modeling/character_cnn.py"""
        super(Highway, self).__init__()
        self.in_D = in_D
        self.activation_func = F.leaky_relu

        # Define Highway (2 layers)
        self.layer1 = nn.Linear(in_D, in_D*2)
        self.layer2 = nn.Linear(in_D, in_D*2)

        # Set bias so that inputs are carried forward
        self.layer1.bias[in_D:].data.fill_(1)
        self.layer2.bias[in_D:].data.fill_(1)


    def forward(self, X_tok_embd):

        # Unrolling to avoid for loop
        # Layer 1
        cur_X_tok_embd = X_tok_embd
        proj_X_tok_embd = self.layer1(cur_X_tok_embd)
        linear_part = cur_X_tok_embd
        nonlinear_part, gate = proj_X_tok_embd.chunk(2, dim=-1)
        nonlinear_part = self.activation_func(nonlinear_part)
        gate = torch.sigmoid(gate)
        cur_X_tok_embd = gate * linear_part + (1 - gate) * nonlinear_part

        # Layer 2
        proj_X_tok_embd = self.layer2(cur_X_tok_embd)
        linear_part = cur_X_tok_embd
        nonlinear_part, gate = proj_X_tok_embd.chunk(2, dim=-1)
        nonlinear_part = self.activation_func(nonlinear_part)
        gate = torch.sigmoid(gate)
        cur_X_tok_embd = gate * linear_part + (1 - gate) * nonlinear_part
        return cur_X_tok_embd


class CharCNN(nn.Module):

    def __init__(self, D, n_chars, max_chars, PAD_idx):
        """CharCNN class adapted from https://github.com/helboukkouri/character-bert/blob/main/modeling/character_cnn.py"""

        super(CharCNN, self).__init__()
        self.D = D
        self.n_chars = n_chars
        self.max_chars = max_chars
        self.PAD_idx = PAD_idx

        self.activation_func = F.leaky_relu
        self.filters = [[1,32], [2,32], [3,32], [4,64], [5,64], [6,128], [7,128]]
        self.n_highways = 2
        self.char_D = 16

        # Initialize character embeddings
        weights = np.zeros((self.n_chars+1, self.char_D), dtype="float32") # (n_chars+1, embd_dim)
        self.char_embd_weights = nn.Parameter(torch.FloatTensor(weights))

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=self.char_D, out_channels=self.filters[0][1], kernel_size=self.filters[0][0])
        self.conv2 = nn.Conv1d(in_channels=self.char_D, out_channels=self.filters[1][1], kernel_size=self.filters[1][0])
        self.conv3 = nn.Conv1d(in_channels=self.char_D, out_channels=self.filters[2][1], kernel_size=self.filters[2][0])
        self.conv4 = nn.Conv1d(in_channels=self.char_D, out_channels=self.filters[3][1], kernel_size=self.filters[3][0])
        self.conv5 = nn.Conv1d(in_channels=self.char_D, out_channels=self.filters[4][1], kernel_size=self.filters[4][0])
        self.conv6 = nn.Conv1d(in_channels=self.char_D, out_channels=self.filters[5][1], kernel_size=self.filters[5][0])
        self.conv7 = nn.Conv1d(in_channels=self.char_D, out_channels=self.filters[6][1], kernel_size=self.filters[6][0])

        # Initialize highway
        n_filters = sum(f[1] for f in self.filters)
        self.highways = Highway(n_filters)

        # Initialize projection
        self.projection = nn.Linear(n_filters, self.D, bias=True)


    def forward(self, X_scan):

        # Get batch size
        B = X_scan.shape[0]

        # Embed X_scan
        X_scan_flat = X_scan.reshape(-1, self.max_chars) # (B*(A*L+1), max_chars)
        X_char_embd = F.embedding(X_scan_flat, self.char_embd_weights) # (B*(A*L+1), max_chars, char_D)
        X_char_embd = torch.transpose(X_char_embd, 1, 2) # (B*(A*L+1), char_D, max_chars)

        # Apply convolutions - unrolled to avoid for loop
        X_conv_1 = self.activation_func(torch.max(self.conv1(X_char_embd), dim=2)[0])
        X_conv_2 = self.activation_func(torch.max(self.conv2(X_char_embd), dim=2)[0])
        X_conv_3 = self.activation_func(torch.max(self.conv3(X_char_embd), dim=2)[0])
        X_conv_4 = self.activation_func(torch.max(self.conv4(X_char_embd), dim=2)[0])
        X_conv_5 = self.activation_func(torch.max(self.conv5(X_char_embd), dim=2)[0])
        X_conv_6 = self.activation_func(torch.max(self.conv6(X_char_embd), dim=2)[0])
        X_conv_7 = self.activation_func(torch.max(self.conv7(X_char_embd), dim=2)[0])

        # Apply highway and projection layers
        X_tok_embd = torch.cat([X_conv_1, X_conv_2, X_conv_3, X_conv_4, X_conv_5, X_conv_6, X_conv_7], dim=1) # (B*(A*L+1), ?)
        X_tok_embd = self.highways(X_tok_embd) # (B*(A*L+1), ?)
        X_tok_embd = self.projection(X_tok_embd)# (B*(A*L+1), D)
        X_tok_embd = X_tok_embd.reshape(B, -1, self.D) # (B, A*L+1, D)
        return X_tok_embd
