#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:11:18 2019

@author: Niels Verleysen

Main script for text prediction with recurrent neural networks
This script can be used to train and load models and use these to generate text from scratch.
Currently supported models: From now on I only use LSTM-networks
"""

import argparse
import sys

from LSTM import TextLSTM

class TextPredictor(object):

    def __init__(self):
        description = R'''Main script for text prediction with recurrent neural networks
                        This script can be used to train and load models and use these to generate text from scratch.
                        Currently supported models: From now on I only use LSTM-networks'''

        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('command', choices=['train', 'produce'], help='Subcommand to run. See main.py SUBCOMMAND -h for more info.')

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    """
        Basic step to set up the required model, it can then be trained or used for prediction. Architecture defined by description argument.
    """
    def build_model(self, architecture, dataset):
        model = TextLSTM()

        architecture_list = architecture.split("|")
        model.process_input(dataset)
        model.build(architecture_list)
        return model

    def train(self):
        description = R'''
        Description: 
            Build a new model and train it on a given dataset.
        Example:
            python main.py train "./datasets/Grimm_Fairy_Tales.txt" "300|300|300" -e 15 -b 100
        '''
        parser = self.build_parser(description)
        parser.add_argument('-e', '--epochs', help='Number of training epochs, default is 50', default=50)
        parser.add_argument('-b', '--batch_size', help='Batch size for training, default is 200', default=200)
        args = parser.parse_args(sys.argv[2:])

        model = self.build_model(args.architecture, args.dataset)
        model.train(int(args.epochs), int(args.batch_size))

    def produce(self):
        description = R'''
        Description: 
            Build a model, load the weights and generate text in a given file.
        Example:
            python main.py produce "./Datasets/Grimm_Fairy_Tales.txt" "300|300|300" "./Weights/lstm-weights.hdf5" "generated.txt" "1000"
        '''
        parser = self.build_parser(description)
        parser.add_argument('weights', help='Weights to be loaded. If you have no weights yet, you can train them using the train command. The weights will then be saved in the ./Weights folder.', default="./Weights/lstm-weights-names.hdf5")
        parser.add_argument('output', help='Name of the output file. You can find your output in this file in the Results folder after running the command', default="default_output.txt")
        parser.add_argument('characters', help='Number of characters to produce', default=1000)
        args = parser.parse_args(sys.argv[2:])

        model = self.build_model(args.architecture, args.dataset)
        model.load(args.weights)
        model.generate(int(args.characters), args.output)

    """
        Build basic parser used in train and produce
    """
    def build_parser(self, description):
        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('dataset', help='Path to textfile containing the training data, this is necessary for determining possible output characters')
        parser.add_argument('architecture', help='Architecture of the model, defined by number of nodes in a layer and multiple layers are split by | character')
        return parser


if __name__ == '__main__':
    TextPredictor()
