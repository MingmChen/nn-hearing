# The Cuda Club
# 12/15/16
# Create a pickle of a pandas data frame
# with columns for the encoding of a .wav file,
# a corresponding mel spectrogram, and target for neural network training.
# To be used with cnn_advanced to recognize speaker voices

import librosa
import pandas
import pickle
import numpy as np
import scipy.sparse as sparse
from os import listdir

# base path for our sound files
SOUND_BASE_PATH = "/tdata/Clear/"
PICKLE_NAME = "Clear_advanced.pickle"
# array of all files in path
files = listdir(SOUND_BASE_PATH)
# to hold array of .wav file data for each file (array of arrays)
y_arr = []
# to hold array of flattened mel spectrograms for each file (array of arrays) 
mels_flatten_arr = []
#hold array of targets (what we want to be training on)
target_arr = []
len_arr = []
for filename in files:
	target = filename[5:7]
	target_arr.append(target)
	# y=numpy array, sr=sample rate
	y, sr = librosa.load(SOUND_BASE_PATH + filename)	
	y_arr.append(y)
	# create a mel spectrogram of .wav file and flatten it
	mels_flatten = librosa.feature.melspectrogram(y=y, sr=sr).flatten()
	mels_flatten_arr.append(mels_flatten)
	len_arr.append(len(mels_flatten))

# prepare arrays to be stored as one element per file 
target_arr = np.array(target_arr)
y_arr = np.array(y_arr)
mels_flatten_arr = np.array(mels_flatten_arr, dtype = object)
y_list = y_arr.tolist()
mels_flatten_list = mels_flatten_arr.tolist()

# store .wav file data in a data frame
# The data frame should have a data column containing numpy array and a target
data = {'target':target_arr,'data': y_list, "mels_flatten":mels_flatten_list,"lens":len_arr}
df = pandas.DataFrame(data)

#get minimum spectorgram length
min_len = df["lens"].min()
del df["lens"] #delete unnecessary column

#Make all spectrograms the same length
df["mels_flatten"] = df.mels_flatten.apply(lambda mels: mels[0:min_len])

#serialize data table to file
df.to_pickle(PICKLE_NAME)
