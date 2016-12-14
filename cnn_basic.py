import librosa
import pandas
import pickle
import numpy as np
import math

# read in our data
ds = pandas.read_pickle("NH_basic.pickle")
ds_dim = ds.shape # tuple: (numrows, numcols)
num_rows = ds_dim[0]

# encode one-hot vector based on row
num_targets = ds_dim[0]
min_target = ds.loc[:,'target'].min()
def to1hot(row):
	one_hot = np.zeros(num_targets)
	one_hot[row - min_target] = 1.0
	return one_hot

ds["one_hot_encoding"] = ds.target.apply(to1hot)

# Form training, testing, and validation data sets
train_data = ds[0:(num_rows - 5)]
validation_data = ds[(num_rows - 5):]
test_data = ds[0:(num_rows - 5)]

# found with  ds.loc[0,'mels_flatten'].shape
INPUT_SHAPE = (136, 32)

train_x = np.vstack(train_data.mels_flatten).reshape(train_data.shape[0], INPUT_SHAPE[0], INPUT_SHAPE[1],1).astype(np.float32)
train_y = np.vstack(train_data["one_hot_encoding"])
train_size = train_y.shape[0]
