"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import tensorflow as tf

class Model(object):
	def __init__(self):
		
		self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
		self.y_input = tf.placeholder(tf.int64, shape = [None])

		self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

		# first convolutional layer
		self.W_conv1 = self._weight_variable([5,5,1,32])
		self.b_conv1 = self._bias_variable([32])

		self.h_conv1 = tf.nn.relu(self._conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1 = self._max_pool_2x2(self.h_conv1)

		# second convolutional layer
		self.W_conv2 = self._weight_variable([5,5,32,64])
		self.b_conv2 = self._bias_variable([64])

		self.h_conv2 = tf.nn.relu(self._conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
		self.h_pool2 = self._max_pool_2x2(self.h_conv2)

		# first fully connected layer
		self.W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
		self.b_fc1 = self._bias_variable([1024])

		h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)
		self.h_fc1 = h_fc1
		# output layer
		self.W_fc2 = self._weight_variable([1024,10])
		self.b_fc2 = self._bias_variable([10])

		self.pre_softmax = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2

		self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=self.y_input, logits=self.pre_softmax)

		self.xent = tf.reduce_sum(self.y_xent)

		self.y_pred = tf.argmax(self.pre_softmax, 1)

		self.correct_prediction = tf.equal(self.y_pred, self.y_input)

		self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		_, self.accuracy_op =\
				tf.metrics.accuracy(labels=self.y_input,\
				predictions=self.y_pred)
		self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

	
		label_mask = tf.one_hot(self.y_input,
							10,
							on_value=1.0,
							off_value=0.0,
							dtype=tf.float32)
		correct_logit = tf.reduce_sum(label_mask * self.pre_softmax, axis=1)
		wrong_logit = tf.reduce_max((1-label_mask) * self.pre_softmax-10000*label_mask, axis=1)
		cw_loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
		self.cw_grad = tf.gradients(cw_loss, self.x_input)[0]

		pgd_loss = self.xent
		self.pgd_grad = tf.gradients(pgd_loss, self.x_input)[0]

		self.blob_grad = tf.gradients(self.y_xent, self.x_input)[0]




	@staticmethod
	def _weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	@staticmethod
	def _bias_variable(shape):
		initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial)

	@staticmethod
	def _conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

	@staticmethod
	def _max_pool_2x2(x):
		return tf.nn.max_pool(x,
							ksize = [1,2,2,1],
							strides=[1,2,2,1],
							padding='SAME')




	def tf_propagate1(self, center, error_range, equation, error_row, bias_row):
		#(784,512)*(100,784,1)->(100,784,512)->(100,512)
		l1 = tf.reduce_sum(tf.abs(equation)*tf.expand_dims(error_range, axis=-1), axis=1)
		upper = center+l1+bias_row+tf.reduce_sum(error_row*\
				tf.cast(tf.greater(error_row,0), tf.float32),axis=1)

		lower = center-l1+bias_row+tf.reduce_sum(error_row*\
				tf.cast(tf.less(error_row,0), tf.float32), axis=1)
		
		appr_condition = tf.cast(tf.logical_and(tf.less(lower,0),tf.greater(upper,0)), tf.float32)
		mask = appr_condition*((upper)/(upper-lower+0.000001))
		mask = mask + 1 - appr_condition
		mask = mask*tf.cast(tf.greater(upper, 0), tf.float32)

		bias_row = bias_row*mask
		center = center*mask
		#center += bias_row

		#mask=(100,1,512)
		mask = tf.expand_dims(mask,axis=1)
		#(784,512)*(100,1,512)
		equation = equation*mask
		#(1,512)*(100,1,512)


		I = tf.eye(tf.shape(mask)[-1], dtype=tf.float32)
		
		error_row = tf.concat([error_row,\
					tf.expand_dims(tf.negative(lower), axis=1)*\
					I*tf.expand_dims(appr_condition, axis=1)], axis=1)
		
		error_row = error_row*mask
		
		'''
		zero_constant = tf.constant(0, dtype=tf.float32)
		bool_mask = tf.not_equal(tf.reduce_sum(tf.abs(error_row), axis=-1), zero_constant)
		#bool_mask = tf.expand_dims(bool_mask, axis=-1)
		#tmp = bool_mask
		error_row = tf.boolean_mask(error_row, bool_mask, axis=1)
		
		#tmp = error_row
		'''
		return upper, lower, center, equation, error_row, bias_row


	def tf_interval1(self, estep, batch_size, config):
		upper_input = tf.clip_by_value(self.x_input+estep, 0.0, 1.0)
		lower_input = tf.clip_by_value(self.x_input-estep, 0.0, 1.0)
		error_range = (upper_input-lower_input)/2.0
		center = (lower_input+upper_input)/2
		center = tf.reshape(center, [-1, 28, 28, 1])
		m = tf.shape(self.x_input)[0]
		equation = tf.eye(784, dtype=tf.float32)
		equation = tf.reshape(equation, [784, 28,28,1])
		bias_row = tf.zeros([1,28,28,1], dtype=tf.float32)
		error_row = tf.zeros([m,28,28,1], dtype=tf.float32)
		tmp = center

		self.W_conv1 = self._weight_variable([5,5,1,32])
		self.b_conv1 = self._bias_variable([32])
		return 
		'''
		conv1 = tf.layers.conv2d(
				inputs = center,
				filters = 16,
				kernel_size = [5,5],
				strides = (2,2),
				padding= "same",
				activation=tf.nn.relu,
				name="conv1", reuse=True)

		tmp = conv1
		
		conv2 = tf.layers.conv2d(
				 inputs = conv1,
				 filters = 32,
				 kernel_size = [5,5],
				 strides = (2,2),
				 padding= "same",
				 activation=tf.nn.relu,
				 name="conv2", reuse=True)

		conv2_flat = tf.reshape(conv2,[-1,7*7*32])
		tmp = conv2_flat
		
		h_fc1 = tf.layers.dense(inputs=conv2_flat, units=100,\
					activation=tf.nn.relu, name="fc1", reuse = True)
		tmp = h_fc1
		
		tmp = tf.layers.dense(inputs=h_fc1, units= 10,\
					activation=None, name="pre_softmax", reuse = True)
		'''

		center = tf.layers.conv2d(
					inputs = center,
                    filters = 16,
					kernel_size = [5,5],
					strides = (2,2),
					padding= "same",
					activation=None,
					name="conv1",
					reuse=True,
					use_bias=False)


		center = tf.layers.conv2d(
					inputs = self.x_image,
                    filters = 16,
					kernel_size = [5,5],
					strides = (2,2),
					padding= "same",
					activation=None,
					name="conv1",
					reuse=True,
					use_bias=False)

		equation = tf.layers.conv2d(
					inputs = equation,
                    filters = 16,
					kernel_size = [5,5],
					strides = (2,2),
					padding= "same",
					activation=None,
					name="conv1",
					reuse=True,
					use_bias=False)

		bias_row = tf.layers.conv2d(
					inputs = bias_row,
                    filters = 16,
					kernel_size = [5,5],
					strides = (2,2),
					padding= "same",
					activation=None,
					name="conv1",
					reuse=True,
					use_bias=True)

		error_row = tf.layers.conv2d(
					inputs = error_row,
                    filters = 16,
					kernel_size = [5,5],
					strides = (2,2),
					padding= "same",
					activation=None,
					name="conv1",
					reuse=True,
					use_bias=False)

		tmp = equation
		center_shape = tf.shape(center)
		equation_shape = tf.shape(equation)
		bias_shape = tf.shape(bias_row)
		error_shape = tf.shape(error_row)
		
		# n is the flattened size of hidden layer
		n = center_shape[1]*center_shape[2]*center_shape[3]

		center = tf.reshape(center, [m, n])
		equation = tf.reshape(equation,[784, n])
		error_row = tf.reshape(error_row, [m, 1, n])
		bias_row = tf.reshape(bias_row, [1, n])
		
		upper, lower, center, equation, error_row, bias_row=\
				self.tf_propagate1(center, error_range, equation, error_row, bias_row)

		
		# batch_size, 14,14,16
		center = tf.reshape(center, center_shape)
		# batch_size * 784, 14,14,16
		equation = tf.reshape(equation,[m*equation_shape[0],\
					equation_shape[1], equation_shape[2], 16])
		# batch_size, 14,14,16
		bias_row = tf.reshape(bias_row, [m, bias_shape[1],\
					bias_shape[2], 16])
		# batch_size *(3136+1), 14,14,16
		error_row = tf.reshape(error_row, [m*\
					tf.shape(error_row)[1],error_shape[1],\
					error_shape[2], 16])
		
		center = tf.layers.conv2d(
					inputs = center,
                    filters = 32,
					kernel_size = [5,5],
					strides = (2,2),
					padding= "same",
					activation=None,
					name="conv2",
					reuse=True,
					use_bias=False)

		equation = tf.layers.conv2d(
					inputs = equation,
                    filters = 32,
					kernel_size = [5,5],
					strides = (2,2),
					padding= "same",
					activation=None,
					name="conv2",
					reuse=True,
					use_bias=False)

		bias_row = tf.layers.conv2d(
					inputs = bias_row,
                    filters = 32,
					kernel_size = [5,5],
					strides = (2,2),
					padding= "same",
					activation=None,
					name="conv2",
					reuse=True,
					use_bias=True)

		error_row = tf.layers.conv2d(
					inputs = error_row,
                    filters = 32,
					kernel_size = [5,5],
					strides = (2,2),
					padding= "same",
					activation=None,
					name="conv2",
					reuse=True,
					use_bias=False)

		equation_shape = tf.shape(equation)
		bias_shape = tf.shape(bias_row)
		error_shape = tf.shape(error_row)
		center_shape = tf.shape(center)
		# n is the flattened size
		n = 7*7*32
		
		center = tf.reshape(center, [m, n])
		equation = tf.reshape(equation,[m, 784, n])
		error_row = tf.reshape(error_row, [m, tf.cast(error_shape[0]/m, tf.int32), n])
		bias_row = tf.reshape(bias_row, [m, n])
		
		upper, lower, center, equation, error_row, bias_row =\
				self.tf_propagate1(center, error_range, equation, error_row, bias_row)
		#tmp = center
		
		center = tf.layers.dense(inputs=center, units=100,\
					activation=None, name="fc1", reuse=True,\
					use_bias=False)
		
		equation = tf.layers.dense(inputs=equation, units=100,\
					activation=None, name="fc1", reuse=True,\
					use_bias=False)
		bias_row = tf.layers.dense(inputs=bias_row, units=100,\
					activation=None, name="fc1", reuse=True,\
					use_bias=True)
		error_row = tf.layers.dense(inputs=error_row, units=100,\
					activation=None, name="fc1", reuse=True,\
					use_bias=False)

		upper, lower, center, equation, error_row, bias_row =\
					self.tf_propagate1(center, error_range, equation, error_row, bias_row)

		
		center = tf.layers.dense(inputs=center, units=10, activation=None,\
							name="pre_softmax", reuse=True, use_bias = False)
		equation = tf.layers.dense(inputs=equation, units=10, activation=None,\
							name="pre_softmax", reuse=True, use_bias = False)
		bias_row = tf.layers.dense(inputs=bias_row, units=10, activation=None,\
							name="pre_softmax", reuse=True)
		error_row = tf.layers.dense(inputs=error_row, units=10, activation=None,\
							name="pre_softmax", reuse=True, use_bias = False)
		#tmp = error_row
		
		# normalized the output

		center_t = center[0:1, self.y_input[0]]
		equation_t = equation[0:1,:,self.y_input[0]]
		bias_row_t = bias_row[0:1, self.y_input[0]]
		error_row_t = error_row[0:1, :, self.y_input[0]]
		
		for i in range(1,batch_size):
			center_t = tf.concat([center_t, center[i:i+1, self.y_input[i]]], axis=0)
			equation_t = tf.concat([equation_t, equation[i:i+1,:, self.y_input[i]]], axis=0)
			bias_row_t = tf.concat([bias_row_t, bias_row[i:i+1, self.y_input[i]]], axis=0)
			error_row_t = tf.concat([error_row_t, error_row[i:i+1,:, self.y_input[i]]], axis=0)

		center = center-tf.expand_dims(center_t, axis=-1)
		equation = equation-tf.expand_dims(equation_t, axis=-1)
		bias_row = bias_row-tf.expand_dims(bias_row_t, axis=-1)
		error_row = error_row-tf.expand_dims(error_row_t, axis=-1)

		#tmp = center
		upper, lower, center, equation, error_row, bias_row =\
					self.tf_propagate1(center, error_range, equation, error_row, bias_row)
		
		
		
		
		self.equation = equation
		self.bias_row = bias_row
		self.error_row = error_row

		
		return upper, lower, estep_decay, tmp


