Assignment 3

Dataset not included in this zip

Please place the following files inside SUB_Echo:
FileListSUB_Val.csv
FileListSUB_Train.csv
FileListSUB_Test.csv

Please remove the file from SUB_Echo:
FileListSUB.csv (the original .csv holding the EF values and video filenames) in SUB_Echo

The only changes needed should be to model.py
data.py extracts the samples for testing/training/validating
loss.py contains the MSLE loss function for Q2
modelsource.py contains the source code for r2plus1d_18, to which the supervision was added for Q3
model1.py is the same as model.py, except some minor changes to suit the new model in Q3

To run, simply run the model.py file

Trained on Google COLAB
Please let me know if there are any difficulties