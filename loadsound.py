import librosa
import pandas
import pickle
import numpy as np
import scipy.sparse as sparse
from os import listdir

# base path for our sound files
SOUND_BASE_PATH = "/tdata/stimuli/NH/"
PICKLE_NAME = "NH_basic.pickle"
# array of all files in path
files = listdir(SOUND_BASE_PATH)
# to hold array of .wav file data for each file (array of arrays)
y_arr = []
# to hold array of flattened mel spectrograms for each file (array of arrays) 
mels_flatten_arr = []
#hold array of targets (what we want to be training on)
target_arr = []
for filename in files:
	target = int(filename[2:5])
	target_arr.append(target)
	# y=numpy array, sr=sample rate
	y, sr = librosa.load(SOUND_BASE_PATH + filename)	
	y_arr.append(y)
	# create a mel spectrogram of .wav file and flatten it
	mels_flatten = librosa.feature.melspectrogram(y=y, sr=sr).flatten()
	mels_flatten_arr.append(mels_flatten)

# prepare arrays to be stored as one element per file 
y_arr = np.array(y_arr)
mels_flatten_arr = np.array(mels_flatten_arr)
y_list = y_arr.tolist()
mels_flatten_list = mels_flatten_arr.tolist()

# store .wav file data in a data frame
# The data frame should have a data column containing numpy array and a target
data = {'target':target_arr,'data': y_list, "mels_flatten":mels_flatten_list}
df2 = pandas.DataFrame(data)

#serialize data table to file
df2.to_pickle(PICKLE_NAME)
