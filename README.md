# Text generation with RNN's
Repository to play around with different RNN architectures for my thesis on temporal modeling for safer spacecrafts. I'm using python and tensorflow/keras.

I have picked this project back up after finishing my thesis.

## How to use
The main.py file must be run from the commandline and requires either the command train or produce. The train command is used to learn weights for the given model to predict characters in a given textfile. The produce command can be used to generate text with the model which uses the weights from a given file. (This file can be produced by the train command)

More information can be found by using the option -h when running the script or when running either command.
```
python main.py -h
python main.py train -h
python main.py produce -h
```

To use version three you can use the following commands for training and generation.
```
python main.py train "./datasets/Grimm_Fairy_Tales.txt" "256|256|256|256"
python main.py produce "./Datasets/Grimm_Fairy_Tales.txt" "256|256|256|256" "./Weights/lstm-weights-v3.hdf5" "generated.txt" "5000" "corrected.txt"
```

## Datasets and results
Free to use books can be found on the internet, for example: https://www.gutenberg.org/
These should be downloaded as txt documents and the header should be removed. There are two examples in the Datasets folder.
