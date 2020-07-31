#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:54:07 2019

@author: Niels Verleysen

Predicting text, trained on given textfile
"""

"""Imports"""
import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


class TextLSTM(object):

    def __init__(self):
        self.preprocessed = False
        self.network_ready = False

    """
        Preprocess the given text for training and generating
    """
    def process_input(self, file):
        self.raw_text = open(file, 'r', encoding='utf-8').read()
        self.raw_text = self.raw_text.lower()
        chars = sorted(list(set(self.raw_text)))
        self.char_to_int = dict((c, i) for i, c in enumerate(chars)) #Necessary for training
        self.int_to_char = dict((i, c) for i, c in enumerate(chars)) #Necessary for generation
        
        """Summarize"""
        self.n_chars = len(self.raw_text)
        self.n_vocab = len(chars)
        print("Total Characters: ", self.n_chars)
        print("Total Vocab: ", self.n_vocab)
        
        """Prepare the dataset of input to output pairs encoded as integers"""
        self.seq_length = 100
        self.dataX = []
        dataY = []
        for i in range(0, self.n_chars - self.seq_length, 1):
            seq_in = self.raw_text[i:i + self.seq_length]
            seq_out = self.raw_text[i + self.seq_length]
            self.dataX.append([self.char_to_int[char] for char in seq_in])
            dataY.append(self.char_to_int[seq_out])
            
        """Reshape X to be [samples, time steps, features]"""
        X = np.reshape(self.dataX, (len(self.dataX), self.seq_length, 1))
        """Normalize the data"""
        self.X = X / float(self.n_vocab)
        """One hot encode the output variable"""
        self.Y = np_utils.to_categorical(dataY)

        self.preprocessed = True

    """
        Construct network with LSTM layers based on the architecture list
    """
    def build(self, architecture):
        inputs = Input(shape=(self.X.shape[1], self.X.shape[2]))
        l = inputs
        for layer in architecture:
            l = LSTM(int(layer), dropout=0.02, recurrent_dropout=0.02, return_sequences=True)(l)
        fl = Flatten()(l)
        outputs = Dense(self.Y.shape[1], activation='softmax')(fl)  # Next character
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        
        self.model.summary()
        
        filepath="./Weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        self.callbacks_list = [checkpoint]

        self.network_ready = True

    """
        Train the model
        process_input and build must have been run before trying this
    """
    def train(self, epochs=15, batch_size=200):
        if self.preprocessed and self.network_ready:
            self.model.fit(self.X, self.Y, epochs=epochs, batch_size=batch_size, callbacks=self.callbacks_list)

    """
        Load pretrained model
        build function must have been run before trying this
    """
    def load(self, file):
        if self.network_ready:
            self.model.load_weights(file)

    """
        Generate new text with the network
        process_input and build must have been run before trying this
    """
    def generate(self, size, name='generated_text.txt'):
        if self.preprocessed and self.network_ready:
            start = np.random.randint(0, len(self.dataX)-1)
            pattern = self.dataX[start]

            file = './Results/' + str(name)
            f = open(file, "w+")

            seed = ''.join([self.int_to_char[value] for value in pattern])
            f.write(seed)

            for i in range(size):
                x = np.reshape(pattern, (1, len(pattern), 1))
                x = x / float(self.n_vocab)
                prediction = self.model.predict(x, verbose=0)
                index = np.argmax(prediction)
                result = self.int_to_char[index]
                f.write(result)
                pattern.append(index)
                pattern = pattern[1:len(pattern)]

            print("\n Done")
            f.close()
