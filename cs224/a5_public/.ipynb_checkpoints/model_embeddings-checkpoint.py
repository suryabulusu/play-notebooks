#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
#         pad_token_idx = vocab.src['<pad>']
#         self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        
        pad_token_idx = vocab.char2id['<pad>']
        self.char_emb_size = 50
        self.word_emb_size = embed_size
        self.embeddings = nn.Embedding(len(vocab.char2id), self.char_emb_size, padding_idx = pad_token_idx)
        
        
        self.dropout = nn.Dropout(p = 0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        
        
        x_emb = self.embeddings(input) # (sentence_length, batch_size, max_word_length, char_emb_length)
        x_reshaped = torch.transpose(x_emb, 2, 3)
        
        
        cnn = CNN(char_emb_len = self.char_emb_size, 
                  word_emb_len = self.word_emb_size, 
                  max_word_len = x_reshaped.shape[-1])
        
#         print("before", x_reshaped.shape)
        
        x_conv_out = cnn(x_reshaped.view(-1, self.char_emb_size, x_reshaped.shape[-1]))
        
#         print(x_conv_out.shape)
        
        highway = Highway(word_emb_length = self.word_emb_size)
        
        x_highway = highway(x_conv_out)
        
        x_word_emb = self.dropout(x_highway)
        
        return x_word_emb.view(x_reshaped.shape[0], x_reshaped.shape[1], self.word_emb_size)

        ### END YOUR CODE

