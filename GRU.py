#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:16:26 2019

@author: Niels Verleysen

Predicting text, trained on given textfile
"""

"""Imports"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, GRU, Input
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


class TextGRU(object):
    
    def process_input(self, file):
        self.raw_text = open(file, 'r', encoding='utf-8').read()
        chars = sorted(list(set(self.raw_text)))
        self.char_to_int = dict((c, i) for i, c in enumerate(chars)) #Necessary for training
        self.int_to_char = dict((i, c) for i, c in enumerate(chars)) #Necessary for generation
        
        """Summarize"""
        self.n_chars = len(self.raw_text)
        self.n_vocab = len(chars)
        print("Total Characters: ", self.n_chars)
        print("Total Vocab: ", self.n_vocab)
        
        """Prepare the dataset of input to output pairs encoded as integers"""
        seq_length = 7
        self.dataX = []
        dataY = []
        for i in range(0, self.n_chars - seq_length, 1):
            	seq_in = self.raw_text[i:i + seq_length]
            	seq_out = self.raw_text[i + seq_length]
            	self.dataX.append([self.char_to_int[char] for char in seq_in])
            	dataY.append(self.char_to_int[seq_out])
            
        """Reshape X to be [samples, time steps, features]"""
        X = np.reshape(self.dataX, (len(self.dataX), seq_length, 1))
        """Normalize the data"""
        self.X = X / float(self.n_vocab)
        """One hot encode the output variable"""
        self.Y = np_utils.to_categorical(dataY)
    
    def build(self, architecture):
        inputs = Input(shape=(self.X.shape[1], self.X.shape[2]))
        previous_layer = inputs
        for layer in architecture:
            next_layer = GRU(layer, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(previous_layer)
            previous_layer = next_layer
        outputs = Dense(self.Y.shape[1], activation='softmax')(previous_layer) # Next character
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        
        filepath="./Weights/gru-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        self.callbacks_list = [checkpoint]
        
    def train(self, epochs=15, batch_size=200):
        self.model.fit(self.X, self.Y, epochs=epochs, batch_size=batch_size, callbacks=self.callbacks_list)
    
    def load(self, file):
        self.model.load_weights(file)
    
    def generate(self, size, name='generated_text.txt'):
        start = np.random.randint(0, len(self.dataX)-1)
        pattern = self.dataX[start]
        
        file = './Results/' + str(name)
        f= open(file,"w+")
        
        for i in range(size):
            	x = np.reshape(pattern, (1, len(pattern), 1))
            	x = x / float(self.n_vocab)
            	prediction = self.model.predict(x, verbose=0)
            	index = np.argmax(prediction)
            	result = self.int_to_char[index]
            	seq_in = [self.int_to_char[value] for value in pattern]
            	f.write(result)
            	pattern.append(index)
            	pattern = pattern[1:len(pattern)]
        
        print("\n Done")
        f.close()