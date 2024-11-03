#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
1d Convolution Layer
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """ 1d Convolution + MaxPooling Model
        - Takes a word's character embeddings
        - Gives a word embedding by combining char embs
    """
    def __init__(self, char_emb_len, word_emb_len, max_word_len, kernel_size = 5):
        """ Initializes CNN
        @param char_emb_len (int): Character Embedding Length
        @param word_emb_len (int): Word Embedding Length
        @param max_word_len (int): Max Word Length (21)
        @param kernel_size (int): Default value 5
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels = char_emb_len, 
                                out_channels = word_emb_len, 
                                kernel_size = kernel_size, 
                                bias = True)
        
        self.maxpool = nn.MaxPool1d(kernel_size = max_word_len - kernel_size + 1)
        
    def forward(self, x_reshaped):
        """
        @param x_reshaped (Tensor): shape (batch_size, char_emb_len, max_word_len)
        
        @returns x_conv_out (Tensor): shape (batch_size, word_emb_len)
        """
        x_conv = self.conv(x_reshaped)
        
        x_conv_out = self.maxpool(F.relu(x_conv))
        
        x_conv_out = torch.squeeze(x_conv_out, dim = -1) # very dangerous line, always specify dimension
        
        return x_conv_out
        
### END YOUR CODE

