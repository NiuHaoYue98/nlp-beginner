# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 13:00:48 2020

Task 2 - textCNN Model 

@author: apple
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

class textCNN(nn.Module):
    def __init__(self,vocab_size,vocab_dim,kernel_num,kernel_list,class_num,dropout_rate,embedding_matrix):
        super(textCNN,self).__init__()

        in_channel = 1
        
        self.embed = nn.Embedding(vocab_size,vocab_dim)
        #self.embed.weight.data.copy_(embedding_matrix)
        #self.embed.weight.require_grad=False #使词向量在训练中保持固定


        self.convs = nn.ModuleList([nn.Conv2d(in_channel,kernel_num,(k,vocab_dim)) for k in kernel_list])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(kernel_list)*kernel_num,class_num)
        
    def forward(self,x):
        # x.shape = [batch_size,fix_length]
        x = x.t()
        embedding_x = self.embed(x)   # embedding_x.shape = [batch_size,fix_length,vocab_dim]
        x = embedding_x.unsqueeze(1)  # x.shape = [batch_size,in_channel,fix_length,vocab_dim]
        # 使用不同的卷积核进行卷积
        conv_out = [F.relu(conv(x)).squeeze(3) for conv in self.convs] #len(kernel_list) * kernel_num
        # 池化操作
        max_pool_out = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in conv_out] # len(kernel_list) * (N,kernel_num)
        # 向量拼接
        out = torch.cat(max_pool_out,1) #(N,kernel_num*len(kernel_list))
        out = self.dropout(out)
        y = self.fc(out) # (N,C)
        return y
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
