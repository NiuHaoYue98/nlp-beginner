# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 18:40:21 2020

@author: apple
"""

import numpy as np
import matplotlib.pyplot as plt

class SoftmaxRegression():
    def __init__(self,X,Y,alpha,epoch):
        self.X = X
        self.Y = Y
        self.batchnum = len(X)
        self.batchsize,self.features = X[0].shape
        self.classnum = Y.shape[2]
        
        self.W = np.ones((self.classnum,self.features))
        self.alpha = alpha
        self.epoch = epoch
        self.loss = []
        
    def softmax(self,Y):
        # 【需要减去每列的最大值，否则求exp(x)会溢出】
        y_max = Y.max(axis=0)
        y_exp = np.exp(Y - y_max)
        y_sum = np.sum(y_exp,axis=0,keepdims=True)
        y = y_exp / y_sum
        return y
        
    def train(self):
        print('Start Training')
        for i in range(self.epoch):
            round_loss = 0
            for batch in range(self.batchnum):
                batch_x = self.X[batch]
                batch_y = self.Y[batch].T
                pred_y = np.dot(self.W,batch_x.T)
                softmax_y = self.softmax(pred_y)
                loss = 0
                for j in range(self.batchsize):
                    loss += np.dot((batch_y[:,j]-softmax_y[:,j]).reshape(self.classnum,1),batch_x[j].T.reshape(1,self.features))
                loss = loss / self.batchsize
                self.W = self.W + self.alpha * loss 
                round_loss += np.sum((batch_y-softmax_y)**2)
            self.loss.append(round_loss)
            print('Round ',i,'Loss',round_loss)
        plt.plot(range(len(self.loss)),self.loss)
        np.savetxt('W.txt',self.W)
            
    def predict(self,test_X):
        pred_y = np.dot(self.W,test_X.T)
        softmax_y = self.softmax(pred_y)
        out = np.argmax(softmax_y,axis=0)
        return out
    
    def load_para(self):
        self.W = np.loadtxt('W.txt')
        
        
        
        
        
        
        
        
        
        
        
        
        