import librosa
import pandas
import pickle
import numpy as np

num_targets = 0

#todo: convert this into a loop for getting all files and storing in a df
#Current target: word file ID (NOT the speaker ID)
filename = "PB021M1NAT.wav"
target = int(filename[2:5])
min_target = target
num_targets += 1

# y=numpy array, sr=sample rate
y, sr = librosa.load("/tdata/stimuli/NH/" + filename)

# generate a mel spectogram from the numpy array of the .wav file
mels = librosa.feature.melspectrogram(y=y, sr=sr)

# store .wav file data in a data frame
# The data frame should have a data column containing numpy array and a target
ds = pandas.DataFrame({"data":y,"target":target, "mels": mels}) 

def to1hot(row):
	one_hot = np.zeros(num_targets)
	one_hot[row - min_target] = 1.0
	return one_hot

ds["one_hot_encoding"] = ds.target.apply(to1hot)


