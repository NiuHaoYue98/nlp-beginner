# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:22:54 2020

Bag-of-word
@author: apple
"""

import numpy as np
import pandas as pd

class BagOfWord():
    def __init__(self):
        self.vocab = set()
        self.vocab_size = 0
    
    def gen_vocab(self,X):
        for batch in X:
            for phrase in batch:
                items = set(phrase.split(' '))
                self.vocab = self.vocab | items
        self.vocab = self.vocab | {'<UNK>'}
        self.vocab = dict(zip(list(self.vocab),list(range(len(self.vocab)))))
        self.vocab_size = len(self.vocab)
        print('Vocab Size: ',self.vocab_size)
        self.save_vocab()
        
    def gen_feature(self,X):
        features = []
        for batch in X:
            batch_feature = np.zeros((len(batch),self.vocab_size))
            for i,phrase in enumerate(batch):
                words = phrase.split(' ')
                for word in words:
                    if word not in self.vocab.keys():
                        batch_feature[i][self.vocab['<UNK>']] += 1
                    else:
                        batch_feature[i][self.vocab[word]] += 1
            features.append(batch_feature)
        return features
    
    def save_vocab(self):
        out = pd.DataFrame()
        out['Word'] = list(self.vocab.keys())
        out['Index'] = list(self.vocab.values())
        out.to_csv('bow.vocab',index=False)
    
    def load_vocab(self):
        load_vocab = pd.read_csv('bow.vocab')
        self.vocab = dict(zip(load_vocab['Word'],load_vocab['Index']))
        self.vocab_size = len(self.vocab)
        
        