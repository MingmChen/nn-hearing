import librosa
import pandas
import pickle
import numpy as np
import math
import tensorflow as tf

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

mels_len = ds.loc[0,'mels_flatten'].shape[0]
x = tf.placeholder(tf.float32,[None, mels_len])

W = tf.Variable(tf.zeros([mels_len, num_targets]))
b = tf.Variable(tf.zeros([num_targets]))

y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, num_targets])


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

x_dat = np.vstack(ds.mels_flatten).astype(np.float32)
y_dat = np.vstack(ds.one_hot_encoding).astype(np.float32) 

# train
for _ in range(1000):
	batch_xs = x_dat
	batch_ys = y_dat
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# test the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: x_dat, y_: y_dat}))

# Form training, testing, and validation data sets
#train_data = ds[0:(num_rows - 5)]
#validation_data = ds[(num_rows - 5):]
#test_data = ds[0:(num_rows - 5)]

# found with  ds.loc[0,'mels_flatten'].shape
#INPUT_SHAPE = (136, 32)

#train_x = np.vstack(train_data.mels_flatten)#.reshape(train_data.shape[0], INPUT_SHAPE[0], INPUT_SHAPE[1],1).astype(np.float32)
#train_y = np.vstack(train_data["one_hot_encoding"])
#train_size = train_y.shape[0]
