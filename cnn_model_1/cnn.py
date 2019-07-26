from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from textwrap import wrap
import re
import itertools
import matplotlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import scikitplot as skplt
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import platform

# set constants
TOTAL_EPOCHS = 10
label_count = 10
data_width = 28
data_height = 28
batch_size = 100
learning_rate = 1e-4   # The optimization initial learning rate
num_hidden_units = 1024  # Number of hidden units of the RNN
display_freq = 100      # Frequency of displaying the training results
total_train_data = None
total_test_data = None
log_dir = os.getcwd()
generic_slash = None
if platform.system() == 'Windows':
  generic_slash = '\\'
else:
  generic_slash = '/'

def encodeLabels(labels_decoded):
    encoded_labels = np.zeros(shape=(len(labels_decoded), label_count), dtype=np.int8)
    for x in range(0, len(labels_decoded)):
        some_label = labels_decoded[x]

        if 0 == some_label:
            encoded_labels[x][0] = 1
        elif 1 == some_label:
            encoded_labels[x][1] = 1
        elif 2 == some_label:
            encoded_labels[x][2] = 1
        elif 3 == some_label:
            encoded_labels[x][3] = 1
        elif 4 == some_label:
            encoded_labels[x][4] = 1
        elif 5 == some_label:
            encoded_labels[x][5] = 1
        elif 6 == some_label:
            encoded_labels[x][6] = 1
        elif 7 == some_label:
            encoded_labels[x][7] = 1
        elif 8 == some_label:
            encoded_labels[x][8] = 1
        elif 9 == some_label:
            encoded_labels[x][9] = 1
    return encoded_labels

def weight_variable(shape):
  # uses default std. deviation
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  # uses default bias
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def shallow_cnn(x):
  # Reshape to use within a convolutional neural net
  x_image = tf.reshape(x, [-1, data_width, data_height, 1])

  # Convolutional Layer #1
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  
  h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, 
      strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

  # Pooling Layer #2
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Convolutional Layer #2
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  
  h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, 
      strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

  # Pooling Layer #2
  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Dense Layer
  W_fc1 = weight_variable([7 * 7 * 64, num_hidden_units])
  b_fc1 = bias_variable([num_hidden_units])
  
  # Flatten
  h_pool1_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
  
  # Dropout
  keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Logits Layer
  W_fc2 = weight_variable([num_hidden_units, label_count])
  b_fc2 = bias_variable([label_count])

  y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name = 'y_conv')

  return y_conv, keep_prob

# Load training and eval data
print("Data Loading")
mnist = tf.keras.datasets.mnist
(train_x, train_y),(test_x, test_y) = mnist.load_data()
train_x, test_x = train_x / 255.0, test_x / 255.0

total_train_data = len(train_y)
total_test_data = len(test_y)

print("Encoding Labels")
# One-Hot encode the labels
train_y = encodeLabels(train_y)
test_y = encodeLabels(test_y)

print("Size of:")
print("- Training-set:\t\t{}".format(total_train_data))
print("- Validation-set:\t{}".format(total_test_data))

print("Creating Datasets")
# Create the DATASETs
train_x_dataset = tf.data.Dataset.from_tensor_slices(train_x)
train_y_dataset = tf.data.Dataset.from_tensor_slices(train_y)
test_x_dataset = tf.data.Dataset.from_tensor_slices(test_x)
test_y_dataset = tf.data.Dataset.from_tensor_slices(test_y)

print("Zipping The Data Together")
# Zip the data and batch it and (shuffle)
train_data = tf.data.Dataset.zip((train_x_dataset, train_y_dataset)).shuffle(buffer_size=total_train_data).repeat().batch(batch_size).prefetch(buffer_size=5)
test_data = tf.data.Dataset.zip((test_x_dataset, test_y_dataset)).shuffle(buffer_size=total_test_data).repeat().batch(batch_size).prefetch(buffer_size=1)

print("Creating Iterators")
# Create Iterators
train_iterator = train_data.make_initializable_iterator()
test_iterator = test_data.make_initializable_iterator()

# Create iterator operation
train_next_element = train_iterator.get_next()
test_next_element = test_iterator.get_next()

print("Defining Model Placeholders")
# Create the model
x = tf.placeholder(tf.float32, [None, data_width, data_height], name='x')

# Define loss and optimizer
y_ = tf.placeholder(tf.int8, [None, label_count], name='y_')

# Build the graph for the deep net
y_conv, keep_prob = shallow_cnn(x)

# Create loss op
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
tf.summary.scalar('cross_entropy', cross_entropy)

# Create train op
train_step = tf.train.AdamOptimizer(learning_rate, name='Adam-op').minimize(cross_entropy)

CNN_prediction_label = tf.argmax(y_conv, 1, name='predictions')
actual_label = tf.argmax(y_, 1)
correct_prediction = tf.equal(CNN_prediction_label, actual_label, name='correct_pred')

# Create accuracy op
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Initialize and Run
with tf.Session() as sess:
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(log_dir + generic_slash + 'tensorflow' + generic_slash + 'train', sess.graph)
  test_writer = tf.summary.FileWriter(log_dir + generic_slash + 'tensorflow' + generic_slash + 'test')
  sess.run(tf.global_variables_initializer())
  sess.run(train_iterator.initializer)
  sess.run(test_iterator.initializer)
  saver = tf.train.Saver()
  print("----------------------|----\---|-----|----/----\---|-----|---|\----|---------------------")
  print("----------------------|    |---|     ----|     -------|------|-\---|---------------------")
  print("----------------------|   |----|-----|---|   ---------|------|--\--|---------------------")
  print("----------------------|    |---|     ----|     |------|------|---\-|---------------------")
  print("----------------------|----/---|-----|----\----/---|-----|---|----\|---------------------")
  global_counter = 0
  # Number of training iterations in each epoch
  num_tr_iter = int(total_train_data / batch_size)
  for epoch in range(TOTAL_EPOCHS):
    for iteration in range(num_tr_iter):
      print("step " + str(global_counter))
      batch = sess.run(train_next_element)
      summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      train_writer.add_summary(summary, global_counter)
      train_writer.flush()
      global_counter += 1
  
    # Run validation after every epoch
    validation_batch = sess.run(test_next_element)
    summary, acc = sess.run([merged, accuracy], feed_dict={x: validation_batch[0], y_: validation_batch[1], keep_prob: 1.0})
    print('epoch ' + str(epoch+1) + ', test accuracy ' + str(acc))
    # Save the model
    saver.save(sess, log_dir + generic_slash + "tensorflow" + generic_slash + "mnist_model.ckpt")
    # Save the summaries
    test_writer.add_summary(summary, global_counter)
    test_writer.flush()

  # Evaluate over the entire test dataset
  # Re-initialize
  test_data = tf.data.Dataset.zip((test_x_dataset, test_y_dataset)).shuffle(buffer_size=total_test_data).repeat().batch(total_test_data).prefetch(buffer_size=1)
  test_iterator = test_data.make_initializable_iterator()
  test_next_element = test_iterator.get_next()
  sess.run(test_iterator.initializer)
  
  # Run for final accuracy
  validation_batch = sess.run(test_next_element)

  print('Final Accuracy ' + str(accuracy.eval(feed_dict={
        x: validation_batch[0], y_: validation_batch[1], keep_prob: 1.0})))
  print("FINISHED")
    
  # Re-initialize
  test_data = tf.data.Dataset.zip((test_x_dataset, test_y_dataset)).shuffle(buffer_size=total_test_data).repeat().batch(total_test_data).prefetch(buffer_size=1)
  test_iterator = test_data.make_initializable_iterator()
  test_next_element = test_iterator.get_next()
  sess.run(test_iterator.initializer)

  print("Creating Confusion Matrix")
  validation_batch = sess.run(test_next_element)

  predict, correct = sess.run([CNN_prediction_label, actual_label], feed_dict={
        x: validation_batch[0], y_: validation_batch[1], keep_prob: 1.0})
  skplt.metrics.plot_confusion_matrix(correct, predict, normalize=True)
  plt.savefig(log_dir + generic_slash + "tensorflow" + generic_slash + "plot.png")

