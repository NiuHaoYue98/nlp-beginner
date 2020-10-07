# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 12:47:10 2020

NLP Beginner Task2 

@author: apple
"""

import pandas as pd
import numpy as np

import torchtext.data as data
from torchtext.vocab import GloVe
from torchtext.data import BucketIterator,Iterator
from textCNN import textCNN
from RNN import RNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import matplotlib.pyplot as plt

def load_data(filename,test_filename,shuffle,dataflag):
    if dataflag:

        train_data = pd.read_csv('./data/train.data')
        valid_data = pd.read_csv('./data/valid.data')
#==============================================================================
#         train_data = pd.read_csv('./data/small_train.data')
#         valid_data = pd.read_csv('./data/small_test.data')
#==============================================================================
        test_data = pd.read_csv(test_filename,encoding='utf-8',sep='\t')
    else:
        data = pd.read_csv(filename,encoding='utf-8',sep='\t')
        data = data[['Phrase','Sentiment']].values
        if shuffle:
            np.random.shuffle(data)
        #train_data = data[:15000]
        #test_data = data[15000:20000]
        
        pd.DataFrame(train_data,columns=['Phrase','Sentiment']).to_csv('./data/small_train.data',index=False)
        pd.DataFrame(test_data,columns=['Phrase','Sentiment']).to_csv('./data/small_test.data',index=False)
    return train_data,valid_data,test_data


def word_embedding(train_data,valid_data,test_data,vocab_dim,batchsize):
    TEXT = data.Field(sequential=True,lower=False,fix_length=15,tokenize = lambda x: x.split())
    LABEL = data.Field(sequential=False,use_vocab=False)
    fields = [('id',None),('Phrase',TEXT),('Sentiment',LABEL)]

    train_examples = []
    valid_examples = []
    test_examples = []
    for text,label in zip(train_data['Phrase'],train_data['Sentiment']):
        train_examples.append(data.Example.fromlist([None,text,label],fields))
    for text,label in zip(valid_data['Phrase'],valid_data['Sentiment']):
        valid_examples.append(data.Example.fromlist([None,text,label],fields))
    for textid,text in zip(test_data['PhraseId'],test_data['Phrase']):
        test_examples.append(data.Example.fromlist([textid,text,-1],fields))
    train = data.Dataset(train_examples,fields)
    valid = data.Dataset(valid_examples,fields)
    test = data.Dataset(test_examples,fields)

    TEXT.build_vocab(train,vectors=GloVe(name='6B', dim=vocab_dim))
    embedding_matrix = TEXT.vocab.vectors

    vocab_size = len(TEXT.vocab)
    # 【Step-4】 构建迭代器
    train_iter = BucketIterator(train,batch_size=batchsize)
    valid_iter = BucketIterator(valid,batch_size=batchsize)
    test_iter = Iterator(test,batch_size=batchsize,sort=False, shuffle=False,sort_within_batch=False, repeat=False,train=False)
    return train_iter,valid_iter,test_iter,vocab_size,embedding_matrix


if __name__ == '__main__':
    
    # 【Load Data】
    train_filename = './data/train.tsv/train.tsv'
    test_filename = './data/test.tsv/test.tsv'
    shuffle = True
    dataflag = True
    class_num = 5
    train_data,valid_data,test_data = load_data(train_filename,test_filename,shuffle,dataflag)   #pandas
    train_sample = len(train_data)
    valid_sample = len(valid_data)
    

    # 【Word Embedding】
    vocab_dim = 50
    batch_size = 128
    #embedding_method = int(input('输入词嵌入方法：1表示random embedding, 2表示pre-trained glove embedding : '))
    train_iter,valid_iter,test_iter,vocab_size,embedding_matrix = word_embedding(train_data,valid_data,test_data,vocab_dim,batch_size)
    
    # 【Classification Model】
    model_type = int(input('输入文本分类模型：1表示text-CNN，2表示RNN：'))
    # 1 textCNN
    if model_type == 1:
        print('使用【text-CNN】训练模型')
        kernel_list = [3,4,5]
        kernel_num = 100
        dropout_rate = 0.5
        lr = 0.01
        model_file = 'TextCNN-Random.pt'

        model = textCNN(vocab_size,vocab_dim,kernel_num,kernel_list,class_num,dropout_rate,embedding_matrix)
    # 2 RNN
    elif model_type == 2:
        print('使用【RNN】训练模型')
        hidden_size = 32
        lr = 0.001
        model_file = 'TextRNN-Random.pt'
        model = RNN(vocab_size,vocab_dim,batch_size,hidden_size,class_num,embedding_matrix)
    
    
    # train
    print('模型开始训练……')
    epoch = 50
    loss_list = []
    acc_list = []
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    for i in range(epoch):
        correct_train = 0
        epoch_loss = 0
        for j,batch in enumerate(train_iter):
            x = batch.Phrase   
            y = batch.Sentiment # shape [64]
            #print(i,j,x,y)
            
            optimizer.zero_grad()
    
            y_pred = model(x)
            correct_train += (torch.max(y_pred,1)[1] == y.data).sum()
            #pred_y =y_pred.argmax(dim=1).detach().numpy()
            loss = F.cross_entropy(y_pred,y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        loss_list.append(epoch_loss)
        print(i,epoch_loss)
        accuracy = 100 * correct_train.item() / train_sample
        acc_list.append(accuracy)
        print('Accuracy in the train set is: ',accuracy)
    
    plt.plot(range(len(loss_list)),loss_list)
    plt.title('Loss Curve in Train Set')
    plt.show()
    plt.plot(range(len(acc_list)),acc_list)
    plt.title('Accuracy Curve in Train Set')
    torch.save(model,model_file)
    
    
#==============================================================================
#     # 【Evaluation】
#     # trainset evaluation
#     #model = torch.load(model_file)
#     correct_train = 0
#     for batch in train_iter:
#         x = batch.Phrase
#         y = batch.Sentiment
#         y_pred = model(x)
#         #print(y_pred.shape)
#         correct_train += (torch.max(y_pred,1)[1] == y.data).sum()
#         #correct_train += (torch.max(y_pred,1)[1].view(y.size()).data == y.data).sum()
#     print('Correct num in trainset is: ',correct_train.item())
#     accuracy = 100 * correct_train.item() / train_sample
#     print('Accuracy in the train set is: ',accuracy)
#==============================================================================
    
    # validset evaluation
    valid_result = []
    correct_valid = 0
    for batch in valid_iter:
        x = batch.Phrase
        y = batch.Sentiment
        y_pred = model(x)
        valid_result += torch.max(y_pred,1)[1].numpy().tolist()
        correct_valid += (torch.max(y_pred,1)[1] == y.data).sum()
        #correct_test += (torch.max(y_pred,1)[1].view(y.size()).data == y.data).sum()
    print('Correct num in test set is: ',correct_valid.item())
    accuracy = 100 * correct_valid.item() / valid_sample
    print('Accuracy in the test set is: ' ,accuracy)
    #pd.DataFrame(valid_result).to_csv('valid_result.csv',index=False)
    
    # 【Output】
    model = torch.load(model_file)
    correct = 0
    result = []
    for batch in test_iter:
        x = batch.Phrase
        y_pred = model(x)
        y_pred = torch.max(y_pred,1)[1].numpy().tolist()
        result += y_pred
    
    test_data['Sentiment'] = result
    test_data[['PhraseId','Sentiment']].to_csv('Result.csv',index=False)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

