# -*- coding: utf-8 -*-
"""
RNN Model 
@author: apple
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self,vocab_size,vocab_dim,batch_size,hidden_size,class_num,embedding_matrix):
        super(RNN,self).__init__()
        
        self.embed = nn.Embedding(vocab_size,vocab_dim)
        #self.embed.weight.data.copy_(embedding_matrix)
        #self.embed.weight.require_grad=False #使词向量在训练中保持固定

        self.batchsize = batch_size
        
        self.rnn = nn.RNN(vocab_dim,hidden_size,num_layers=2,dropout=0.5)
        self.linear = nn.Linear(hidden_size,class_num)
        
        
    def forward(self,x):
        # x.shape [seq_len,batch_size]
        embedding_x = self.embed(x)
        # embedding_x.shape [seq_len,batch_size,vocab_dim]
        output,hidden = self.rnn(embedding_x)
        # output.shape [15,64,128]      hidden.shape [2,64,128]
        out = self.linear(hidden).squeeze(0)[-1,:,:]
        # out.shape [2,64,5]
        #out = self.linear(hidden.view(self.batchsize,-1))
        return out
        
