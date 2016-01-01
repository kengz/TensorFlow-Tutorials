import tensorflow as tf


# matrix1 = tf.constant([[3., 3.]])
# matrix2 = tf.constant([[2.], [2.]])
# product = tf.matmul(matrix1, matrix2)

# # session must be closed
# sess = tf.Session()
# result = sess.run(product)
# print result
# sess.close()

# # equivalently will auto-end session
# with tf.Session() as sess:
# 	result = sess.run([product])
# 	print result
	

# # to specify device
# with tf.Session() as sess:
# 	with tf.device("/cpu:0"):
# 		matrix1 = tf.constant([[3., 3.]])
# 		matrix2 = tf.constant([[2.], [2.]])
# 		product = tf.matmul(matrix1, matrix2)
# 		result = sess.run([product])
# 		print result



# # interactive session to vaoid keep variable holding the sess
# # use Tensor.eval() and Operation.run()
# sess = tf.InteractiveSession()

# x = tf.Variable([1.0, 2.0])
# a = tf.constant([3.0, 3.0])

# # initialize the op x
# x.initializer.run()

# # subtraction op: x-a; eval the result tensor
# sub = tf.sub(x, a)
# print sub.eval()

# sess.close()



# # Variable to maintain state. e.g. as counter:
# state = tf.Variable(0, name="counter")

# one = tf.constant(1)
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)

# # Variable must be init with init()
# init_op = tf.initialize_all_variables()

# with tf.Session() as sess:
# 	sess.run(init_op)
# 	print sess.run(state)

# 	for _ in xrange(3):
# 		sess.run(update)
		# print sess.run(state)



# # Fetch output. pass in the tensor tp run([]) to retrieve. Execution will not be redundant
# input1 = tf.constant(3.0)
# input2 = tf.constant(2.0)
# input3 = tf.constant(5.0)
# intermed = tf.add(input2, input3)
# mul = tf.mul(input1, intermed)

# with tf.Session() as sess:
# 	result = sess.run([mul, intermed])
# 	print result
	

# # Feeds: a tensor directly into comp graph instead of using Constants and Variables
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# output = tf.mul(input1, input2)

# with tf.Session() as sess:
# 	print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))