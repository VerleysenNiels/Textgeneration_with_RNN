# RNN-playground
Repository to play around with different RNN architectures for my thesis on temporal modeling for safer spacecrafts. I'm using python and tensorflow/keras.

## How to use
The main.py file must be run from the commandline and requires either the command train or produce. The train command is used to learn weights for the given model to predict characters in a given textfile. The produce command can be used to generate text with the model which uses the weights from a given file. (This file can be produced by the train command)

More information can be found by using the option -h when running the script or when running either command.
```
python main.py -h
python main.py train -h
python main.py produce -h
```

## Datasets and results
Training on the shakespeare dataset takes more time then I can allow at this moment. After training for two days an accuracy of about 30% was reached. The generated text is therefore nothing impressive. A file with 1000 generated characters can be found in the results folder.

Instead used a dataset with names from New York, can be downloaded from data.gov. I processed this dataset to only include names of seven characters or longer. The full dataset and the processed dataset can be found in the datasets folder. All three types of models were then trained on this dataset to test the code from this repository. The results are not incredibly good as I didn't try to find optimal architectures. The generated files can be found in the results folder. 

After transitioning to the main.py module, these older weights don't seem to be able to be loaded in the model anymore.
I'm currently checking if this can be fixed.
