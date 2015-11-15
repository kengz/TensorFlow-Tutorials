import tensorflow as tf
import numpy as np

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



# Simple graph
# mat1 = tf.constant([[3., 3.]])
# mat2 = tf.constant([[2.], [2.]])
# prod = tf.matmul(mat1, mat2)

# sess = tf.Session()
# res = sess.run(prod)
# print res

# sess.close()


# with tf.Session() as sess:
# 	res = sess.run(prod)
# 	print res


# sess.close()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
	with tf.device("/cpu:0"):
		mat1 = tf.constant([[3., 3.]])
		mat2 = tf.constant([[2.], [2.]])
		prod = tf.matmul(mat1, mat2)
		res = sess.run(prod)
		print res

