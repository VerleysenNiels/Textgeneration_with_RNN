#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:11:18 2019

@author: Niels Verleysen

Main script for text prediction with recurrent neural networks
This script can be used to train and load models and use these to generate text from scratch.
Currently supported models: simple RNN, LSTM and GRU.
"""

import argparse
import sys

from LSTM import TextLSTM
from GRU import TextGRU
from RNN import TextRNN

class TextPredictor(object):
   
    def __init__(self):
        description = R'''Main script for text prediction with recurrent neural networks
                        This script can be used to train and load models and use these to generate text from scratch.
                        Currently supported models: simple RNN, LSTM and GRU.'''
        
        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('command', choices=['train', 'produce'], help='Subcommand to run. See main.py SUBCOMMAND -h for more info.')
        
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        
        getattr(self, args.command)()
    
    def build_model(self, model_type, architecture, dataset):
        '''Basic step to set up the required model, it can then be trained or used for prediction. Architecture defined by description argument.'''
        model = None
        if model_type == "SimpleRNN":
            model = TextRNN()
        elif model_type == "LSTM":
            model = TextLSTM()
        elif model_type == "GRU":
            model = TextGRU()
        else:
            print("Given model type is not supported. Supported model types are SimpleRNN, LSTM and GRU.")
            exit(1)
            
        architecture_list = architecture.split("|")
        model.process_input(dataset)
        model.build(architecture_list)    
        return model
    
    def train(self):
        description = R'''
        Description: 
            Build a new model and train it on a given dataset.
        Example:
            python main.py train "./datasets/NY_long_names.txt" "LSTM" "512|512" -e 15 -b 100
        '''
        parser = self.build_parser(description)
        parser.add_argument('-e', '--epochs', help='Number of training epochs, default is 15', default=15)
        parser.add_argument('-b', '--batch_size', help='Batch size for training, defautl is 100', default=100)
        args = parser.parse_args(sys.argv[2:])
        
        model = self.build_model(args.model_type, args.architecture, args.dataset)
        model.train(int(args.epochs), int(args.batch_size))
        
    def produce(self):
        description = R'''
        Description: 
            Build a model, load the weights and generate text in a given file.
        Example:
            python main.py produce "./Datasets/NY_long_names.txt" "LSTM" "512|512" "./Weights/lstm-weights-names.hdf5" "example_names.txt" "50"
        '''
        parser = self.build_parser(description)        
        parser.add_argument('weights', help='Weights to be loaded. If you have no weights yet, you can train them using the train command. The weights will then be saved in the ./Weights folder.')
        parser.add_argument('output', help='Name of the output file. You can find your output in this file in the Results folder after running the command')
        parser.add_argument('characters', help='Number of characters to produce')
        args = parser.parse_args(sys.argv[2:])
        
        model = self.build_model(args.model_type, args.architecture, args.dataset)
        model.load(args.weights)
        model.generate(int(args.characters), args.output)
        
    def build_parser(self, description):
        '''Build basic parser used in train and produce'''
        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('dataset', help='Path to textfile containing the training data, this is necessary for determining possible output characters')
        parser.add_argument('model_type', help='Type of layers to be used; Supported types are: SimpleRNN, GRU and LSTM')
        parser.add_argument('architecture', help='Architecture of the model, defined by number of nodes in a layer and multiple layers are split by | character')
        return parser
        
if __name__ == '__main__':
    TextPredictor()