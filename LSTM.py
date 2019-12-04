#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:54:07 2019

@author: Niels Verleysen

Predicting text, trained on shakespeare textfile
"""

"""Imports"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('dark_background')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D
from keras.losses import categorical_crossentropy


class LSTM(object):
    
    #ToDo: add argument parser allowing this model to be run from the commandline and allow different architectures and databases
    def __init__(self):
        description = R'''LSTM network for text prediction, trained on given dataset.'''
        
        file = "./Datasets/shakespeare.txt"
        
        """Process input file"""
        self.process_input(file)
        
        """Build model, then train on given file and produce output"""
        self.build()
        self.train()
        self.generate(100)
    
    def process_input(self, file):
        self.raw_text = open(file, 'r', encoding='utf-8').read()
        chars = sorted(list(set(self.raw_text)))
        self.char_to_int = dict((c, i) for i, c in enumerate(chars))
        
        """Summarize"""
        n_chars = len(self.raw_text)
        n_vocab = len(chars)
        print("Total Characters: ", n_chars)
        print("Total Vocab: ", n_vocab)
        
        """Prepare the dataset of input to output pairs encoded as integers"""
        seq_length = 100
        dataX = []
        dataY = []
        for i in range(0, n_chars - seq_length, 1):
            	seq_in = self.raw_text[i:i + seq_length]
            	seq_out = self.raw_text[i + seq_length]
            	dataX.append([self.char_to_int[char] for char in seq_in])
            	dataY.append(self.char_to_int[seq_out])
            
        """Reshape X to be [samples, time steps, features]"""
        X = np.reshape(dataX, (n_patterns, seq_length, 1))
        """Normalize the data"""
        self.X = X / float(n_vocab)
        """One hot encode the output variable"""
        self.Y = np_utils.to_categorical(dataY)
    
    def build(self):
        self.model = Sequential()
        self.model.add(LSTM(512, dropout=0.5, recurrent_dropout=0.5))
        self.model.add(LSTM(512, dropout=0.5, recurrent_dropout=0.5))
        self.model.add(Dense(self.Y.shape[1], activation='softmax')) # Next character
        
        model_lstm.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    
    def train(self, file):
        print("ToDo")
    
    def generate(size):
        print("ToDo")
    
if __name__ == '__main__':
    LSTM()