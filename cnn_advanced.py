# The Cuda Club
# 12/15/16
# Design, train, and evaluate a basic neural netwok.
# This neural network trains of mel spectrogram encodings of .wav files
# labeled with speaker titles in order to learn their voices

import librosa
import pandas
import pickle
import numpy as np
import math
import tensorflow as tf

# read in our data
ds = pandas.read_pickle("Vocoded_advanced.pickle")
ds_dim = ds.shape # tuple: (numrows, numcols)
num_rows = ds_dim[0]

# encode one-hot vector based on row
unique_targets = ds.target.unique().tolist()
num_targets = len(unique_targets)
def to1hot(row):
	one_hot = np.zeros(num_targets)
	one_hot[unique_targets.index(row)] = 1.0
	return one_hot

ds["one_hot_encoding"] = ds.target.apply(to1hot)

mels_len = ds.loc[0,'mels_flatten'].shape[0]

# implement softmax regression model
x = tf.placeholder(tf.float32,[None, mels_len])
W = tf.Variable(tf.zeros([mels_len, num_targets]))
b = tf.Variable(tf.zeros([num_targets]))
y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, num_targets])


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

cutoff = 50
train_x = np.vstack(ds.mels_flatten).astype(np.float32)[0:cutoff]
train_y = np.vstack(ds.one_hot_encoding).astype(np.float32)[0:cutoff] 
test_x = np.vstack(ds.mels_flatten).astype(np.float32)[cutoff:]
test_y = np.vstack(ds.one_hot_encoding).astype(np.float32)[cutoff:] 

# train the model on batches of our spectrograms
for _ in range(1000):
	batch_xs = train_x
	batch_ys = train_y
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# test the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y})) #print accuracy
