#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    """Highway Network
        - based on: https://arxiv.org/abs/1505.00387
        - Srivatsava et al., 2015
    """
    def __init__(self, word_emb_length):
        """ Init Highway Net
        
        @param word_emb_length (int): Length of Word Embedding 
        """
        super(Highway, self).__init__()
        self.W_proj = nn.Linear(word_emb_length, word_emb_length, bias = True)
        self.W_gate = nn.Linear(word_emb_length, word_emb_length, bias = True)
        
    def forward(self, x_conv_out):
        """ Take a batch of word vectors which are outputs of 1d Convolution on character 
        embeddings of a word, and return another vector of same size, but with a highway 
        connection. This helps with gradient flows.
        
        @param x_conv_out (Tensor): shape (batch_size, word_emb_length)
        
        @returns x_highway (Tensor): shape (batch_size, word_emb_length)
        
        """
#         x_proj = nn.ReLU()((self.W_proj(x_conv_out)))

        x_proj = F.relu(self.W_proj(x_conv_out))
        x_gate = torch.sigmoid(self.W_gate(x_conv_out))
        
        
#         print(x_conv_out.shape, x_gate.shape)
        x_highway = torch.einsum("ab, ab -> ab", x_gate, x_proj) + torch.einsum("ab, ab -> ab", 1 - x_gate, x_conv_out)
        
        return x_highway

### END YOUR CODE 

