# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 18:08:26 2020

@author: apple
"""

import numpy as np
import pandas as pd

class NGram():
    def __init__(self,n=2):
        self.vocab = set()
        self.n = n
        self.vocab_size = 0
    
    def gen_vocab(self,X):
        for batch in X:
            for phrase in batch:
                items = phrase.split(' ')
                ngrams = []
                for i in range(1,self.n+1):
                    if len(items) < self.n:
                        ngrams.append(' '.join(items[:]))
                        break
                    for index in range(len(items)-i+1):
                        ngram = ' '.join(items[index:index+i])
                        ngrams.append(ngram)
                ngrams = set(ngrams)
                self.vocab = self.vocab | ngrams
        self.vocab = self.vocab | {'<UNK>'}
        self.vocab = dict(zip(list(self.vocab),list(range(len(self.vocab)))))
        self.vocab_size = len(self.vocab)
        print('Vocab Size: ',self.vocab_size)
        self.save_vocab()
        
    def gen_feature(self,X):
        features = []
        for batch in X:
            batch_feature = np.zeros((len(batch),self.vocab_size),dtype=int)
            for i,phrase in enumerate(batch):
                words = phrase.split(' ')
                for j in range(1,self.n+1):
                    if len(words) < self.n:
                        ngram = ' '.join(words[:])
                        if ngram not in self.vocab.keys():
                             batch_feature[i][self.vocab['<UNK>']] += 1
                        else:
                             batch_feature[i][self.vocab[ngram]] += 1
                        break
                    for index in range(len(words)-j+1):
                        ngram = ' '.join(words[index:index+j])
                        if ngram not in self.vocab.keys():
                            batch_feature[i][self.vocab['<UNK>']] += 1
                        else:
                            batch_feature[i][self.vocab[ngram]] += 1
            features.append(batch_feature)
        return features
    
    def save_vocab(self):
        out = pd.DataFrame()
        out['Word'] = list(self.vocab.keys())
        out['Index'] = list(self.vocab.values())
        out.to_csv('ngram.vocab',index=False)
    
    def load_vocab(self):
        load_vocab = pd.read_csv('ngram.vocab')
        self.vocab = dict(zip(load_vocab['Word'],load_vocab['Index']))
        self.vocab_size = len(self.vocab)
    
    
    
    
    
    
    
    
    
    
    
    
    
