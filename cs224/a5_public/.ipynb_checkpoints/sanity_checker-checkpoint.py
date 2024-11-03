"""
Self-made Sanity Checker for Assignment 5

Usage:
    sanity_checker.py highway
    sanity_checker.py cnn
"""

import numpy as np
from docopt import docopt

import torch
import torch.nn as nn

from highway import Highway
from cnn import CNN

def configure(m):
    
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(1)
        if type(m) == nn.Conv1d:
            m.weight.data.fill_(1)
    
    with torch.no_grad():
        m.apply(init_weights)

def highway_checker():
    model = Highway(word_emb_length = 3)
    
    print("Preparing sanity checker for Highway layer...")
    print("-"*80)
    
    # configure model
    configure(model)
    
    # load expected outputs
    
    test_inp = torch.tensor([[0, 0, 5], [5, 0, 0]], dtype = torch.float)
    
    print(model(test_inp))
    
    print("-"*80)
    print("Sanity checks for Highway layer passed!")
    
def cnn_checker():
    model = CNN(char_emb_len = 3, word_emb_len = 6, max_word_len = 10, kernel_size = 5)
    
    print("Preparing sanity checker for CNN layer...")
    print("-"*80)
    
    # configure model
    configure(model)
    
    test_inp = torch.randn(2, 3, 10) # (batch_size, char_emb_len, max_word_len)
    
#     x_conv, x_conv_out, x_conv_out2 = model(test_inp)
    x_conv_out = model(test_inp)
    
#     print(x_conv.shape)
    
#     print(x_conv_out.shape)
    
    print(x_conv_out.shape)
    print("-"*80)
    print("Sanity checks for CNN Layer passed!")
    

def main():
    args = docopt(__doc__)
    
    if args['highway']:
        highway_checker()
    
    if args['cnn']:
        cnn_checker()
        
if __name__ == "__main__": main()