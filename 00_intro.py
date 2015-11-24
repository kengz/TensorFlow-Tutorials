# source url: http://www.tensorflow.org/get_started/basic_usage.md#the-computation-graph

import tensorflow as tf
import numpy as np
import os

# x_data = np.float32(np.random.rand(2,10))
# y_data = np.dot([0.100, 0.200], x_data) + 0.300
# # print x_data
# # print y_data

# # linear model for zee y_data
# b = tf.Variable(tf.zeros([1]))
# W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
# y = tf.matmul(W, x_data) + b

# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)

# init = tf.initialize_all_variables()

# sess = tf.Session()
# sess.run(init)

# for step in xrange(0, 201):
# 	sess.run(train)
# 	if step % 20:
# 		print step, sess.run(W), sess.run(b)



# Simple graph: below are 3 nodes
# mat1 = tf.constant([[3., 3.]])
# mat2 = tf.constant([[2.], [2.]])
# prod = tf.matmul(mat1, mat2)

# sess = tf.Session()
# res = sess.run(prod)
# print res

# sess.close()

# alternative way to run
# with tf.Session() as sess:
# 	res = sess.run(prod)
# 	print res

# sess.close()

# or specify device, log it
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
# 	with tf.device("/cpu:0"):
# 		mat1 = tf.constant([[3., 3.]])
# 		mat2 = tf.constant([[2.], [2.]])
# 		prod = tf.matmul(mat1, mat2)
# 		res = sess.run(prod)
# 		print res




# InteractiveSession = so don't have to build ur own comp graph
# then u can use Tensor.evel() and Operation.run()
# sess = tf.InteractiveSession()

# x = tf.Variable([1.0, 2.0])
# a = tf.constant([3.0, 3.0])

# # initialize x
# x.initializer.run()
# # add op node, subtraction
# sub = tf.sub(x, a)
# print sub.eval()


# Variables
# create a var init to val 0
# var = tf.Variable(0, name="counter")
# # create an Op to add 1 to var
# one = tf.constant(1)
# new_value = tf.add(var, one)
# update = tf.assign(var, new_value)

# # launch the graph. add the init op first
# init_op = tf.initialize_all_variables()

# # launch the graph n run the ops
# with tf.Session() as sess:
# 	sess.run(init_op)
# 	print sess.run(var)
# 	for _ in range(3):
# 		sess.run(update)
# 		print sess.run(var)



# Fetches
# To fetch the outputs of operations, execute the graph with a run() call on the Session object and pass in the tensors to retrieve. In the previous example we fetched the single node var, but you can also fetch multiple tensors:

# input1 = tf.constant(3.0)
# input2 = tf.constant(2.0)
# input3 = tf.constant(5.0)
# intermed = tf.add(input2, input3)
# mul = tf.mul(input1, intermed)

# with tf.Session() as sess:
# 	# pass in the tensors to retrieve
# 	res = sess.run([mul, intermed])
# 	print res




# Feeds to placeholder()
# replaces input node with external feed
# input1 = tf.placeholder(tf.types.float32)
# input2 = tf.placeholder(tf.types.float32)
# output = tf.mul(input1, input2)

# with tf.Session() as sess:
# 	print sess.run([output], feed_dict={input1:[7.], input2:[2.]})





# # Saving Variables
# note by default saves all Variables,
# u can sepcify some subset by e.g.: saver = tf.train.Saver({"my_v2": v2})
# mat1 = tf.Variable([[2., 3.]], name="m1")
# mat2 = tf.Variable([[0., 1.], [1., 0.]], name="m2")
# op = tf.matmul(mat1, mat2)

# # the op to init all Variables in parallel
# init = tf.initialize_all_variables()

# # the op to save n restore all Variables
# saver = tf.train.Saver()

# with tf.Session() as sess:
# 	sess.run(init)
# 	res = sess.run(op)
# 	print res
# 	# saving to disk
# 	save_path = saver.save(sess, os.getcwd()+"/model.ckpt")
# 	print "Model saved in file: ", save_path



# # Restoring the model
# v1 = tf.Variable([[0., 0.]], name="m1")
# v2 = tf.Variable([[0., 0.], [0., 0.]], name="m2")
# op2 = tf.matmul(v1, v2)

# init = tf.initialize_all_variables()
# saver = tf.train.Saver()

# with tf.Session() as sess:
# 	saver.restore(sess, os.getcwd()+"/model.ckpt")
# 	print "Model restored."
# 	print sess.run(op2)

