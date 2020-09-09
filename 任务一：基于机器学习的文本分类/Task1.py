# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:17:47 2020

Task 1: Text Classification
Method: softmax-regression

@author: apple
"""

import numpy as np
import pandas as pd
from BagOfWord import BagOfWord
from NGram import NGram
from SoftmaxRegression import SoftmaxRegression

def load_data(filename,shuffle,dataflag):
    if dataflag:
        train_data = np.array(pd.read_csv('./data/small_train.data'))
        test_data = np.array(pd.read_csv('./data/small_test.data'))
    else:
        data = pd.read_csv(filename,encoding='utf-8',sep='\t')
        data = data[['Phrase','Sentiment']].values
        if shuffle:
            np.random.shuffle(data)
        train_data = data[:15000]
        test_data = data[15000:20000]
        
        pd.DataFrame(train_data,columns=['Phrase','Sentiment']).to_csv('./data/small_train.data',index=False)
        pd.DataFrame(test_data,columns=['Phrase','Sentiment']).to_csv('./data/small_test.data',index=False)
    return train_data,test_data

def gen_mini_batch(data,batch_size):
    batched_data = []
    sample_num,feature_num = data.shape
    batch_num = sample_num//batch_size
    for i in range(batch_num):
        batched_data.append(data[i*batch_size:(i+1)*batch_size])
    last_batch_size = len(data[(i+1)*batch_size:])
    last_batch = np.append(data[(i+1)*batch_size:],data[:batch_size-last_batch_size],axis=0)
    batched_data.append(last_batch)
    return np.array(batched_data)

def split_dataset(data):
    X = []
    Y = []
    for batch in data:
        X.append(batch[:,0])
        Y.append(batch[:,1])
    return np.array(X),np.array(Y)

def gen_onehot_label(data):
    Y = []
    for batch in data:
        new_Y = np.zeros((len(batch),5))
        for i in range(len(batch)):
            label = batch[i]
            new_Y[i][label] = 1
        Y.append(new_Y)
    return np.array(Y)

if __name__ == "__main__":
    # hyper-parameter
    train_filename = './data/train.tsv/train.tsv'
    test_filename = './data/test.tsv/test.tsv'              
    shuffle = True
    class_num = 5
    batch_size = 128
    dataflag = True
    vocabflag = True
    input_control = True
    while input_control:
        choose_feature = input('输入1使用Bag-of-Word提取特征，输入2使用NGram提取特征：')
        if choose_feature == '1' or choose_feature == '2':
            input_control = False
        else:
            print('输入错误，请重新输入！')
    
    # load file
    train_data,test_data = load_data(train_filename,shuffle,dataflag)
    
    # generate mini-batch
    batched_train_data = gen_mini_batch(train_data,batch_size)
    train_X,train_Y = split_dataset(batched_train_data)
    
    # change label
    train_Y = gen_onehot_label(train_Y) # X[batch_num,batch_size] Y[batch_num,batch_size,feature_num]
    
    # feature extraction
    if choose_feature == '1':
        # 【bow】
        bow = BagOfWord()
        if vocabflag:
            bow.load_vocab()
        else:
            bow.gen_vocab(train_X)
        train_X = bow.gen_feature(train_X)
        print('Bow Input Shape: train_X',len(train_X),train_X[0].shape,'train_Y',train_Y.shape) # X,Y [batch_num,batch_size,feature_num] 
    elif choose_feature == '2':
        # 【N-gram】
        ngram = NGram(n=2)
        if vocabflag:
            ngram.load_vocab()
        else:
            ngram.gen_vocab(train_X)
        train_X = ngram.gen_feature(train_X)
        print('NGram Input Shape: train_X',len(train_X),train_X[0].shape,'train_Y',train_Y.shape) # X,Y [batch_num,batch_size,feature_num] 
    
    # training
    alpha = 0.5
    epoch = 200
    model = SoftmaxRegression(train_X,train_Y,alpha,epoch)
    #model.load_para()          #读取模型参数，方便调试，完成一次训练后才有相应的文件
    model.train()
    correct = 0
    train_batch_num = len(train_X)
    for i in range(train_batch_num):
        train_batch = train_X[i]
        train_pred_Y = model.predict(train_batch).tolist()
        for j in range(batch_size):
            if train_pred_Y[j] == np.argmax(train_Y[i][j],axis=0):
                correct += 1
    print('Accuracy in training set is: ',correct/(batch_size*(train_batch_num-1)))
    
    # evaluation
    test_X = test_data[:,0]
    test_Y = test_data[:,1]
    test_batch = []
    result = []
    test_batch_num = len(test_data)//batch_size+1
    for i in range(test_batch_num):
        #test_batch = test_X[i*batch_size:(i+1)*batch_size]
        test_batch.append(test_X[i*batch_size:(i+1)*batch_size])
    if choose_feature == '1':
        test_X = bow.gen_feature(np.array(test_batch))
    else:
        test_X = ngram.gen_feature(np.array(test_batch))
    for i in range(test_batch_num):
        test_pred_Y = model.predict(test_X[i]).tolist()
        result += test_pred_Y
    correct = 0
    for i in range(len(test_data)):
        if result[i] == test_Y[i]:
            correct += 1
    print('Accuracy in test set is:',correct/len(test_data))
    
    # outputfile
    out_result = []
    out_data = pd.read_csv(test_filename,encoding='utf-8',sep='\t')[40000:]
    out_X = out_data['Phrase'].values
    batch_num = len(out_data)//batch_size+1
    out_batch = []    
    for i in range(batch_num):
        out_batch.append(out_X[i*batch_size:(i+1)*batch_size])
    if choose_feature == '1':
        out_X = bow.gen_feature(np.array(out_batch))
    elif choose_feature == '2':
        out_X = ngram.gen_feature(np.array(out_batch))
    for i in range(batch_num):
        out_pred_Y = model.predict(out_X[i]).tolist()
        out_result += out_pred_Y

    out_data['Sentiment'] = out_result
    out_data[['PhraseId','Sentiment']].to_csv('Result-ngram-2.csv',index=False)
    

    
    



























