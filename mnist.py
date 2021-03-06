# import tensorflow as tf

# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print sess.run(hello)

# a = tf.constant(10)
# b = tf.constant(32)
# print sess.run(a+b)

import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()

# x = tf.placeholder("float", shape=[None, 784])
# y_ = tf.placeholder("float", shape=[None, 10])

# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))

# init = tf.initialize_all_variables()
# sess.run(init)

# # regression model
# y = tf.nn.softmax(tf.matmul(x,W)+b)
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# for i in range(1000):
# 	batch = mnist.train.next_batch(50)
# 	train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels})

# sess.close()



# # With Tensorboard
# x = tf.placeholder("float", shape=[None, 784], name="x-input")
# W = tf.Variable(tf.zeros([784, 10]), name="weights")
# b = tf.Variable(tf.zeros([10]), name="bias")

# # use name scope to organize nodes in the graph visualizer
# with tf.name_scope("Wx_b") as scope:
#   y = tf.nn.softmax(tf.matmul(x, W)+b)

# # Add summary ops to collect input_data
# w_hist = tf.histogram_summary("weights", W)
# b_hist = tf.histogram_summary("biases", b)
# y_hist = tf.histogram_summary("y", y)

# # Define loss n optimizer
# y_ = tf.placeholder("float", shape=[None, 10], name="y-input")
# # even moar name scope:
# with tf.name_scope("xent") as scope:
#   cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#   ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
# with tf.name_scope("train") as scope:
#   train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# with tf.name_scope("test") as scope:
#   correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#   accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#   accuracy_summary = tf.scalar_summary("accuracy", accuracy)
# # Merge all summaries, write to output

# merged = tf.merge_all_summaries()
# writer = tf.train.SummaryWriter("/Users/theredrose/Documents/TensorFlow-Tutorials/logs", sess.graph_def)
# tf.initialize_all_variables().run()

# # Train le model, feed in test data n record summaries every 10 steps
# for i in range(1000):
#   if i%10 == 0:
#     feed = {x: mnist.test.images, y_: mnist.test.labels}
#     result = sess.run([merged, accuracy], feed_dict=feed)
#     summary_str = result[0]
#     acc = result[1]
#     writer.add_summary(summary_str, i)
#     print("Accuracy at step %s: %s" % (i, acc))
#   else:
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     feed = {x: batch_xs, y_:batch_ys}
#     sess.run(train_step, feed_dict=feed)

# print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))



# # old MNIST
# x = tf.placeholder("float", shape=[None, 784], name="x-input")
# y_ = tf.placeholder("float", shape=[None, 10], name="y-input")

# W = tf.Variable(tf.zeros([784,10]), name="weights")
# b = tf.Variable(tf.zeros([10]), name="bias")

# sess.run(tf.initialize_all_variables())


# with tf.name_scope("Wx_b") as scope:
#   y = tf.nn.softmax(tf.matmul(x,W) + b)

# # add summary op to collect input data
# w_hist = tf.histogram_summary("weights", W)
# b_hist = tf.histogram_summary("biases", b)
# y_hist = tf.histogram_summary("y", y)

# with tf.name_scope("xent") as scope:
#   cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#   ce_sum = tf.scalar_summary("cross entropy", cross_entropy)

# with tf.name_scope("train") as scope:
#   train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# # for i in range(1000):
# #   batch = mnist.train.next_batch(50)
# #   train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# with tf.name_scope("test") as scope:
#   correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#   accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#   accuracy_summary = tf.scalar_summary("accuracy", accuracy)


# Deep MNIST
x = tf.placeholder("float", shape=[None, 784], name="x-input")
y_ = tf.placeholder("float", shape=[None, 10], name="y-input")

def weight_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1, name=name)
	return tf.Variable(initial)

def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape, name=name)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

with tf.name_scope("conv1") as scope:
  W_conv1 = weight_variable([5,5,1,32], "W_conv1")
  b_conv1 = bias_variable([32], "b_conv1")
  x_image = tf.reshape(x, [-1,28,28,1])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope("conv2") as scope:
  W_conv2 = weight_variable([5,5,32,64], "W_conv2")
  b_conv2 = bias_variable([64], "b_conv2")
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope("fc1") as scope:
  W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
  b_fc1 = bias_variable([1024], "b_fc1")
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope("fc2") as scope:
  W_fc2 = weight_variable([1024, 10], "W_fc2")
  b_fc2 = bias_variable([10], "b_fc2")

with tf.name_scope("y_conv") as scope:
  y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

w_hist = tf.histogram_summary("weights_conv1", W_conv1)
b_hist = tf.histogram_summary("biases_conv1", b_conv1)
y_hist = tf.histogram_summary("y_conv", y_conv)

with tf.name_scope("xent") as scope:
  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  entropy_summary = tf.scalar_summary("conv_entropy", cross_entropy)

with tf.name_scope("test_conv") as scope:
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  accuracy_summary = tf.scalar_summary("conv_accuracy", accuracy)


merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/Users/theredrose/Documents/TensorFlow-Tutorials/logs", sess.graph_def)


sess.run(tf.initialize_all_variables())

# for i in range(20000):
for i in range(2000):
  batch = mnist.train.next_batch(50)
  if i%50 == 0:
    feed = {x:batch[0], y_: batch[1], keep_prob: 1.0}
    result = sess.run([merged, accuracy], feed_dict=feed)
    summary_str = result[0]
    acc = result[1]
    writer.add_summary(summary_str)
    print "step %d, training accuracy %g"%(i, acc)
  else:
    feed = {x:batch[0], y_: batch[1], keep_prob: 1.0}
    sess.run(train_step, feed_dict=feed)

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
