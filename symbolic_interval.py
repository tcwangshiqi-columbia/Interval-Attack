from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack
import numpy as np


tmp = tf.constant(0)

def tf_propagate1(center, error_range, equation, error_row, bias_row):
	#(784,512)*(100,784,1)->(100,784,512)->(100,512)
	l1 = tf.reduce_sum(tf.abs(equation)*\
			tf.expand_dims(error_range, axis=-1), axis=1)
	upper = center+l1+bias_row+tf.reduce_sum(error_row*\
			tf.cast(tf.greater(error_row,0), tf.float32),axis=1)

	lower = center-l1+bias_row+tf.reduce_sum(error_row*\
			tf.cast(tf.less(error_row,0), tf.float32), axis=1)
	
	appr_condition = tf.cast(tf.logical_and(tf.less(lower,0),\
						tf.greater(upper,0)), tf.float32)
	
	mask = appr_condition*((upper)/(upper-lower+0.000001))
	mask = mask + 1 - appr_condition
	mask = mask*tf.cast(tf.greater(upper, 0), tf.float32)
	
	global tmp
	tmp = mask
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
	
	
	#zero_constant = tf.constant(0, dtype=tf.float32)
	#bool_mask = tf.not_equal(tf.reduce_sum(tf.abs(error_row), axis=0), zero_constant)
	#bool_mask = tf.expand_dims(bool_mask, axis=-1)
	#tmp = bool_mask
	#error_row = tf.boolean_mask(error_row, bool_mask, axis=1)
	#tmp = error_row
	
	return upper, lower, center, equation, error_row, bias_row


def conv1_graph(w_dict):
	tf_x = tf.placeholder(tf.float32, shape = [None, 784])
	tf_x = tf.reshape(tf_x, [-1, 28, 28, 1])
	tf_conv1 = tf.nn.conv2d(tf_x, w_dict["w_conv1"],\
				strides=[1,1,1,1], padding='SAME')+w_dict["b_conv1"]
	tf_conv1 = tf.nn.relu(tf_conv1)
	return tf_x, tf_conv1



def build_tf_graph_whole(w_dict, estep):
	estep_decay = tf.placeholder('float', []) 
	tf_x1 = tf.placeholder(tf.float32, shape = [1, 784])
	tf_y1 = tf.placeholder(tf.int64, shape = [None])
	tf_mask1 = tf.placeholder(tf.bool, shape = [1,28,28,32])
	tf_mask2 = tf.placeholder(tf.bool, shape = [1,14,14,64])
	tf_mask_seq1 = tf.placeholder(tf.int64, shape = [6272])
	tf_mask_seq2 = tf.placeholder(tf.int64, shape = [3136])
	#tf_x1 = tf.reshape(tf_x1, [1, 28, 28, 1])
	upper_input = tf.clip_by_value(tf_x1+estep*estep_decay, 0.0, 1.0)
	lower_input = tf.clip_by_value(tf_x1-estep*estep_decay, 0.0, 1.0)
	error_range = (upper_input-lower_input)/2.0
	center = (lower_input+upper_input)/2
	center = tf.reshape(center, [1, 28, 28, 1])
	m = 1
	equation = tf.eye(784, dtype=tf.float32)
	equation = tf.reshape(equation, [784, 28,28,1])
	bias_row = tf.zeros([1,28,28,1], dtype=tf.float32)
	error_row = tf.zeros([1,28,28,1], dtype=tf.float32)

	center = tf.nn.conv2d(center, w_dict["w_conv1"],\
				strides=[1,1,1,1], padding='SAME')
	equation = tf.nn.conv2d(equation, w_dict["w_conv1"], \
				strides=[1,1,1,1], padding='SAME')
	bias_row = tf.nn.conv2d(bias_row, w_dict["w_conv1"],\
				strides=[1,1,1,1], padding='SAME')+w_dict["b_conv1"]
	error_row = tf.nn.conv2d(error_row, w_dict["w_conv1"],\
				strides=[1,1,1,1], padding='SAME')
	center_shape = tf.shape(center)
	equation_shape = tf.shape(equation)
	bias_shape = tf.shape(bias_row)
	error_shape = tf.shape(error_row)

	n = center_shape[1]*center_shape[2]*center_shape[3]

	center = tf.reshape(center, [1, n])
	equation = tf.reshape(equation,[784, n])
	error_row = tf.reshape(error_row, [1, 1, n])
	bias_row = tf.reshape(bias_row, [1, n])

	upper, lower, center, equation, error_row, bias_row=\
			tf_propagate1(center, error_range,\
			equation, error_row, bias_row)

	# batch_size, 14,14,16
	center = tf.reshape(center, center_shape)
	# batch_size * 784, 14,14,16
	equation = tf.reshape(equation,[1*equation_shape[0],\
				equation_shape[1], equation_shape[2], 32])
	# batch_size, 14,14,16
	bias_row = tf.reshape(bias_row, [1, bias_shape[1],\
				bias_shape[2], 32])
	# batch_size *(3136+1), 14,14,16
	error_row = tf.reshape(error_row, [1*\
				tf.shape(error_row)[1],error_shape[1],\
				error_shape[2], 32])

	
	# max1
	zero_constant = tf.constant(0, dtype=tf.float32)

	center = tf.boolean_mask(center, tf_mask1)
	center = tf.gather(center, tf_mask_seq1)
	center = tf.reshape(center, [1,14,14,32])

	bias_row = tf.boolean_mask(bias_row, tf_mask1)
	bias_row = tf.gather(bias_row, tf_mask_seq1)
	bias_row = tf.reshape(bias_row, [1,14,14,32])

	equation = tf.boolean_mask(equation,\
				tf.reshape(tf_mask1, [28,28,32]), axis=1)
	equation = tf.gather(equation, tf_mask_seq1, axis=1)
	equation = tf.reshape(equation, [784,14,14,32])

	error_n = tf.shape(error_row)[0]

	error_row = tf.boolean_mask(error_row,\
				tf.reshape(tf_mask1, [28,28,32]), axis=1)
	error_row = tf.gather(error_row, tf_mask_seq1, axis=1)
	error_row = tf.reshape(error_row, [error_n, 14,14,32])


	
	# conv2_graph1
	center = tf.nn.conv2d(center, w_dict["w_conv2"],\
				strides=[1,1,1,1], padding='SAME')
	equation = tf.nn.conv2d(equation, w_dict["w_conv2"],\
				strides=[1,1,1,1], padding='SAME')
	bias_row = tf.nn.conv2d(bias_row, w_dict["w_conv2"],\
				strides=[1,1,1,1], padding='SAME')+w_dict["b_conv2"]
	error_row = tf.nn.conv2d(error_row, w_dict["w_conv2"],\
				strides=[1,1,1,1], padding='SAME')
	center_shape = tf.shape(center)
	equation_shape = tf.shape(equation)
	bias_shape = tf.shape(bias_row)
	error_shape = tf.shape(error_row)
	
	n = center_shape[1]*center_shape[2]*center_shape[3]

	center = tf.reshape(center, [1, n])
	equation = tf.reshape(equation,[784, n])
	error_row = tf.reshape(error_row, [1, tf.cast(error_shape[0], tf.int32), n])
	bias_row = tf.reshape(bias_row, [1, n])
	
	upper, lower, center, equation, error_row, bias_row=\
			tf_propagate1(center, error_range, equation, error_row, bias_row)

	# batch_size, 14,14,16
	center = tf.reshape(center, center_shape)
	# batch_size * 784, 14,14,16
	equation = tf.reshape(equation,[1*equation_shape[0],\
				equation_shape[1], equation_shape[2], 64])
	# batch_size, 14,14,16
	bias_row = tf.reshape(bias_row, [1, bias_shape[1],\
				bias_shape[2], 64])
	# batch_size *(3136+1), 14,14,16
	error_row = tf.reshape(error_row, [1*\
				tf.shape(error_row)[1],error_shape[1],\
				error_shape[2], 64])



	#max2
	center = tf.boolean_mask(center, tf_mask2)
	center = tf.gather(center, tf_mask_seq2)
	center = tf.reshape(center, [1,7,7,64])

	bias_row = tf.boolean_mask(bias_row, tf_mask2)
	bias_row = tf.gather(bias_row, tf_mask_seq2)
	bias_row = tf.reshape(bias_row, [1,7,7,64])

	equation = tf.boolean_mask(equation,\
				tf.reshape(tf_mask2, [14,14,64]), axis=1)
	equation = tf.gather(equation, tf_mask_seq2, axis=1)
	equation = tf.reshape(equation, [784,7,7,64])

	error_n = tf.shape(error_row)[0]

	error_row = tf.boolean_mask(error_row,\
				tf.reshape(tf_mask2, [14,14,64]), axis=1)
	error_row = tf.gather(error_row, tf_mask_seq2, axis=1)
	error_row = tf.reshape(error_row, [error_n, 7,7,64])



	# fc layer
	error_shape = tf.shape(error_row)

	m = 1
	n = 7*7*64
	center = tf.reshape(center, [1, n])
	equation = tf.reshape(equation,[784, n])
	error_row = tf.reshape(error_row,\
			[tf.cast(error_shape[0], tf.int32), n])
	bias_row = tf.reshape(bias_row, [1, n])

	center = tf.matmul(center, w_dict["w_fc1"])
	equation = tf.matmul(equation, w_dict["w_fc1"])
	error_row = tf.matmul(error_row, w_dict["w_fc1"])
	bias_row = tf.matmul(bias_row, w_dict["w_fc1"])+w_dict["b_fc1"]

	n = 1024

	center = tf.reshape(center, [1, n])
	equation = tf.reshape(equation,[1, 784, n])
	error_row = tf.reshape(error_row,\
			[1, tf.cast(error_shape[0], tf.int32), n])
	bias_row = tf.reshape(bias_row, [1, n])

	upper, lower, center, equation, error_row, bias_row=\
			tf_propagate1(center, error_range,\
			equation, error_row, bias_row)

	error_shape = tf.shape(error_row)

	center = tf.reshape(center, [1, n])
	equation = tf.reshape(equation,[784, n])
	error_row = tf.reshape(error_row,\
			[tf.cast(error_shape[1], tf.int32), n])
	bias_row = tf.reshape(bias_row, [1, n])

	center = tf.matmul(center, w_dict["w_fc2"])
	equation = tf.matmul(equation, w_dict["w_fc2"])
	error_row = tf.matmul(error_row, w_dict["w_fc2"])
	bias_row = tf.matmul(bias_row, w_dict["w_fc2"])+w_dict["b_fc2"]

	n = 10
	center = tf.reshape(center, [1, n])
	equation = tf.reshape(equation,[1, 784, n])
	error_row = tf.reshape(error_row,\
			[1, tf.cast(error_shape[1], tf.int32), n])
	bias_row = tf.reshape(bias_row, [1, n])
	
	
	center_t = center[0:1, tf_y1[0]]
	equation_t = equation[0:1,:,tf_y1[0]]
	bias_row_t = bias_row[0:1, tf_y1[0]]
	error_row_t = error_row[0:1, :, tf_y1[0]]
	
	for i in range(1,1):
		center_t = tf.concat([center_t,\
				center[i:i+1, tf_y1[i]]], axis=0)
		equation_t = tf.concat([equation_t,\
				equation[i:i+1,:, tf_y1[i]]], axis=0)
		bias_row_t = tf.concat([bias_row_t,\
				bias_row[i:i+1, tf_y1[i]]], axis=0)
		error_row_t = tf.concat([error_row_t,\
				error_row[i:i+1,:, tf_y1[i]]], axis=0)

	center = center-tf.expand_dims(center_t, axis=-1)
	equation = equation-tf.expand_dims(equation_t, axis=-1)
	bias_row = bias_row-tf.expand_dims(bias_row_t, axis=-1)
	error_row = error_row-tf.expand_dims(error_row_t, axis=-1)
	

	#tmp = center
	upper, lower, center, equation, error_row, bias_row =\
				tf_propagate1(center, error_range, equation,\
				error_row, bias_row)


	
	return tf_x1, tf_y1, tf_mask1, tf_mask2, tf_mask_seq1,\
			tf_mask_seq2, estep_decay, upper, lower, center,\
			equation, error_row, bias_row


def conv2_graph(w_dict):
	tf_max1 = tf.placeholder(tf.float32, shape = [None, 14, 14, 32])
	tf_conv2 = tf.nn.conv2d(tf_max1, w_dict["w_conv2"],\
			strides=[1,1,1,1], padding='SAME')+w_dict["b_conv2"]
	tf_conv2 = tf.nn.relu(tf_conv2)
	return tf_max1, tf_conv2


def conv2_graph1(w_dict, estep, error_range):
	centeri = tf.placeholder(tf.float32, shape = [1, 14, 14, 32])
	equationi = tf.placeholder(tf.float32, shape=[784, 14, 14, 32])
	error_rowi = tf.placeholder(tf.float32, shape=[None, 14, 14, 32])
	bias_rowi = tf.placeholder(tf.float32, shape=[1, 14, 14, 32])

	center = tf.nn.conv2d(centeri, w_dict["w_conv2"],\
			strides=[1,1,1,1], padding='SAME')
	equation = tf.nn.conv2d(equationi, w_dict["w_conv2"],\
			strides=[1,1,1,1], padding='SAME')
	bias_row = tf.nn.conv2d(bias_rowi, w_dict["w_conv2"],\
			strides=[1,1,1,1], padding='SAME')+w_dict["b_conv2"]
	error_row = tf.nn.conv2d(error_rowi, w_dict["w_conv2"],\
			strides=[1,1,1,1], padding='SAME')
	center_shape = tf.shape(center)
	equation_shape = tf.shape(equation)
	bias_shape = tf.shape(bias_row)
	error_shape = tf.shape(error_row)

	n = center_shape[1]*center_shape[2]*center_shape[3]

	center = tf.reshape(center, [1, n])
	equation = tf.reshape(equation,[784, n])
	error_row = tf.reshape(error_row, [1, tf.cast(error_shape[0], tf.int32), n])
	bias_row = tf.reshape(bias_row, [1, n])

	upper, lower, center, equation, error_row, bias_row=\
			tf_propagate1(center, error_range, equation, error_row, bias_row)

	# batch_size, 14,14,16
	center = tf.reshape(center, center_shape)
	# batch_size * 784, 14,14,16
	equation = tf.reshape(equation,[1*equation_shape[0],\
				equation_shape[1], equation_shape[2], 64])
	# batch_size, 14,14,16
	bias_row = tf.reshape(bias_row, [1, bias_shape[1],\
				bias_shape[2], 64])
	# batch_size *(3136+1), 14,14,16
	error_row = tf.reshape(error_row, [1*\
				tf.shape(error_row)[1],error_shape[1],\
				error_shape[2], 64])
	return upper, lower, center, equation,\
		error_row, bias_row, centeri, equationi,\
		error_rowi, bias_rowi


def tf_fc(w_dict):
	tf_max2 = tf.placeholder(tf.float32, shape = [None, 7, 7, 64])
	tf_fc1 = tf.reshape(tf_max2, [-1, 3136])
	tf_fc1 = tf.nn.relu(tf.matmul(tf_fc1, w_dict["w_fc1"]) + w_dict["b_fc1"])
	tf_pre_softmax = tf.matmul(tf_fc1, w_dict["w_fc2"]) + w_dict["b_fc2"]
	return tf_max2, tf_pre_softmax


def tf_fc1(w_dict, estep, error_range):
	centeri = tf.placeholder(tf.float32, shape = [1, 7, 7, 64])
	equationi = tf.placeholder(tf.float32, shape=[784, 7, 7, 64])
	error_rowi = tf.placeholder(tf.float32, shape=[None, 7, 7, 64])
	bias_rowi = tf.placeholder(tf.float32, shape=[1, 7, 7, 64])
	tf_y1 = tf.placeholder(tf.int64, shape = [None])

	error_shape = tf.shape(error_rowi)

	m = 1
	n = 7*7*64
	center = tf.reshape(centeri, [1, n])
	equation = tf.reshape(equationi,[784, n])
	error_row = tf.reshape(error_rowi,\
			[tf.cast(error_shape[0], tf.int32), n])
	bias_row = tf.reshape(bias_rowi, [1, n])

	center = tf.matmul(center, w_dict["w_fc1"])
	equation = tf.matmul(equation, w_dict["w_fc1"])
	error_row = tf.matmul(error_row, w_dict["w_fc1"])
	bias_row = tf.matmul(bias_row, w_dict["w_fc1"])+w_dict["b_fc1"]

	n = 1024

	center = tf.reshape(center, [1, n])
	equation = tf.reshape(equation,[1, 784, n])
	error_row = tf.reshape(error_row, [1,\
			tf.cast(error_shape[0], tf.int32), n])
	bias_row = tf.reshape(bias_row, [1, n])

	upper, lower, center, equation, error_row, bias_row=\
			tf_propagate1(center, error_range,\
			equation, error_row, bias_row)

	error_shape = tf.shape(error_row)

	center = tf.reshape(center, [1, n])
	equation = tf.reshape(equation,[784, n])
	error_row = tf.reshape(error_row,\
			[tf.cast(error_shape[1], tf.int32), n])
	bias_row = tf.reshape(bias_row, [1, n])

	center = tf.matmul(center, w_dict["w_fc2"])
	equation = tf.matmul(equation, w_dict["w_fc2"])
	error_row = tf.matmul(error_row, w_dict["w_fc2"])
	bias_row = tf.matmul(bias_row, w_dict["w_fc2"])+w_dict["b_fc2"]

	n = 10
	center = tf.reshape(center, [1, n])
	equation = tf.reshape(equation,[1, 784, n])
	error_row = tf.reshape(error_row,\
			[1, tf.cast(error_shape[1], tf.int32), n])
	bias_row = tf.reshape(bias_row, [1, n])
	
	center_t = center[0:1, tf_y1[0]]
	equation_t = equation[0:1,:,tf_y1[0]]
	bias_row_t = bias_row[0:1, tf_y1[0]]
	error_row_t = error_row[0:1, :, tf_y1[0]]
	
	for i in range(1,1):
		center_t = tf.concat([center_t,\
				center[i:i+1, tf_y1[i]]], axis=0)
		equation_t = tf.concat([equation_t,\
				equation[i:i+1,:, tf_y1[i]]], axis=0)
		bias_row_t = tf.concat([bias_row_t,\
				bias_row[i:i+1, tf_y1[i]]], axis=0)
		error_row_t = tf.concat([error_row_t,\
				error_row[i:i+1,:, tf_y1[i]]], axis=0)

	center = center-tf.expand_dims(center_t, axis=-1)
	equation = equation-tf.expand_dims(equation_t, axis=-1)
	bias_row = bias_row-tf.expand_dims(bias_row_t, axis=-1)
	error_row = error_row-tf.expand_dims(error_row_t, axis=-1)
	
	#tmp = center
	upper, lower, center, equation, error_row, bias_row =\
				tf_propagate1(center, error_range,\
				equation, error_row, bias_row)

	return tf_y1, upper, lower, center, equation,\
		error_row, bias_row, centeri, equationi,\
		error_rowi, bias_rowi








def build_tf_graph(w_dict):
	tf_x, tf_conv1 = conv1_graph(w_dict)
	tf_max1, tf_conv2 = conv2_graph(w_dict)
	tf_max2, tf_pre_softmax = tf_fc(w_dict)
	return tf_x, tf_conv1, tf_max1, tf_conv2, tf_max2, tf_pre_softmax

def build_tf_graph1(w_dict, estep):
	tf_x1, error_range, estep_decay, upper, lower, center,\
				 equation, error_row, bias_row =\
				 conv1_graph1(w_dict, estep)

	param_conv1 = [center, equation, error_row, bias_row]

	upper, lower, center_conv2, equation_conv2, error_row_conv2,\
			bias_row_conv2, center_conv2_input, equation_conv2_input,\
			error_row_conv2_input, bias_row_conv2_input =\
			conv2_graph1(w_dict, estep, error_range)

	param_conv2 = [center_conv2, equation_conv2,\
			error_row_conv2, bias_row_conv2]

	param_conv2_input = [center_conv2_input,\
			equation_conv2_input, error_row_conv2_input,\
			bias_row_conv2_input]
	
	tf_y1, upper, lower, center_fc, equation_fc,\
			error_row_fc, bias_row_fc, center_fc_input,\
			equation_fc_input, error_row_fc_input,\
			bias_row_fc_input = tf_fc1(w_dict, estep, error_range)
	param_fc = [center_fc, equation_fc, error_row_fc, bias_row_fc]
	param_fc_input = [center_fc_input, equation_fc_input,\
			error_row_fc_input, bias_row_fc_input]
	
	return tf_x1, tf_y1, error_range, estep_decay,\
		upper, lower, param_conv1, param_conv2,\
		param_conv2_input, param_fc, param_fc_input

def trasfer_index(a):
	n = a.shape[0]
	b = np.zeros(a.shape)
	for ni in range(n):
		b[a[ni]] = ni
	return b


def np_max(maxi):

	shape = maxi.shape
	max_mask = np.zeros((shape))
	row = shape[1]/2
	col = shape[2]/2
	channel = shape[3]
	maxo = np.zeros((row,col,channel))
	cnt = 1
	'''
	for c in range(channel):
		for i in range(row):
			for j in range(col):
	'''
	for i in range(row):
		for j in range(col):
			for c in range(channel):
				m = -1000
				iim = 0
				jjm = 0
				for ii in range(2):
					for jj in range(2):
						if(maxi[0,2*i+ii,2*j+jj,c]>m):
							m = maxi[0,2*i+ii,2*j+jj,c]
							iim = ii
							jjm = jj

				maxo[i,j,c] = maxi[0,2*i+iim,2*j+jjm,c]

				#maxo[i,j,c] = np.max(maxi[0,2*i:2*i+2,2*j:2*j+2,c])
				max_mask[0,2*i+iim,2*j+jjm,c] = cnt
				cnt += 1


	mask_seq = max_mask[np.nonzero(max_mask)].astype(np.int64)-1
	mask_seq = trasfer_index(mask_seq.reshape(-1)).astype(np.int64)
	'''
	pool = maxi[max_mask!=0]
	pool = pool[mask_seq].reshape(14,14,32)
	
	print maxo[:,:,-1]
	print pool[:,:,-1]
	'''
	return maxo.reshape(1, row, col, channel), max_mask!=0, mask_seq

def np_max2(maxi, equation, error_row, bias_row):
	maxi += bias_row
	shape = maxi.shape
	row = shape[1]/2
	col = shape[2]/2
	channel = shape[3]
	maxo = np.zeros((row,col,channel))
	equationo = np.zeros((784,row,col,channel))
	error_rowo = np.zeros((error_row.shape[0],row,col,channel))
	bias_rowo = np.zeros((1,row,col,channel))

	for c in range(channel):
		for i in range(row):
			for j in range(col):
				m = -1000
				iim = 0
				jjm = 0
				for ii in range(2):
					for jj in range(2):
						if(maxi[0,2*i+ii,2*j+jj,c]>m):
							m = maxi[0,2*i+ii,2*j+jj,c]
							iim = ii
							jjm = jj

				maxo[i,j,c] = maxi[0,2*i+iim,2*j+jjm,c]

				equationo[:,i,j,c] = equation[:,2*i+iim,2*j+jjm,c]
				error_rowo[:,i,j,c] = error_row[:,2*i+iim,2*j+jjm,c]
				bias_rowo[:,i,j,c] = bias_row[:,2*i+iim,2*j+jjm,c]
	centero = maxo-bias_rowo
	return centero, equationo, error_rowo, bias_rowo

'''
def np_max1(maxi, upper, lower, center, equation, error_row, bias_row):
	upper = np.reshape(maxi.shape)*(upper>0)
	lower = np.reshape(maxi.shape)*(lower>0)
	shape = maxi.shape
	row = shape[1]/2
	col = shape[2]/2
	channel = shape[3]
	maxo = np.zeros((row,col,channel))
	equationo = np.zeros((784,row,col,channel))
	centero = np.zeros((1,row,col,channel))
	for c in range(channel):
		for i in range(row):
			for j in range(col):
				u0 = upper[0,2*i,2*j,c]
				l0 = lower[0,2*i,2*j,c]
				u1 = upper[0,2*i+1,2*j,c]
				l1 = lower[0,2*i+1,2*j,c]
				u2 = upper[0,2*i,2*j+1,c]
				l2 = lower[0,2*i,2*j+1,c]
				u3 = upper[0,2*i+1,2*j+1,c]
				l3 = lower[0,2*i+1,2*j+1,c]

				gamma0 = (u0/(u0-l0)+u1/(u1-l1)+u2/(u2-l2)+u3/(u3-l3)-1)/(1/(u0-l0)+1/(u1-l1)+1/(u2-l2)+1/(u3-l3))
				gamma = np.min(np.max(gamma0, np.max(l0,l1,l2,l3)), np.min(u0,u1,u2,u3))

				g0 = (u0-gamma)/(u0-l0)
				g1 = (u1-gamma)/(u1-l1)
				g2 = (u2-gamma)/(u2-l2)
				g3 = (u3-gamma)/(u3-l3)
				G = g0+g1+g2+g3

				if(G<1): eta=np.min(l0,l1,l2,l3)
				elif(G==1): eta=gamma
				else: eta=np.max(u0,u1,u2,u3)

				for k in range(k):
					equationo[k,i,j,c] = g0*equation[k,2*i,2*j,c]+g1*equation[k,2*i+1,2*j,c]+g2*equation[k,2*i,2*j+1,c]+g3*equation[k,2*i+1,2*j+1,c]
				
				for k in range(error_row.shape[1]):
					error_row[k,i,j,c] = g0*error_row[k,2*i,2*j,c]+g1*error_row[k,2*i+1,2*j,c]+g2*error_row[k,2*i,2*j+1,c]+g3*error_row[k,2*i+1,2*j+1,c]


				maxo[i,j,c] = np.max(maxi[0,2*i:2*i+2,2*j:2*j+2,c])
				centero[0,i,j,c] = g0*center[0,2*i,2*j,c]+g1*center[0,2*i+1,2*j,c]+g2*center[0,2*i,2*j+1,c]+g3*center[0,2*i+1,2*j+1,c]
				bias_rowo[0,i,j,c] = g0*bias_rowo[0,2*i,2*j,c]+g1*bias_rowo[0,2*i+1,2*j,c]+g2*bias_rowo[0,2*i,2*j+1,c]+g3*bias_rowo[0,2*i+1,2*j+1,c]

	return centero
'''


def propagate_whole(sess, w_dict, x, y,\
			tf_graph_param_whole, naive_graph, ed):

	o, mask1, mask2, mask_seq1, mask_seq2 =\
			propagate(sess, w_dict, x, naive_graph)
	#print np.sum(o)

	tf_x1, tf_y1, tf_mask1, tf_mask2, tf_mask_seq1,\
			tf_mask_seq2, estep_decay, upper, lower,\
			center, equation, error_row, bias_row = tf_graph_param_whole
	input_dict = {tf_x1:x, tf_y1:y, tf_mask1:mask1,\
			tf_mask2:mask2, tf_mask_seq1:mask_seq1,\
			tf_mask_seq2:mask_seq2, estep_decay:ed}
	c, u, e, l, b = sess.run([center, upper, equation, lower, bias_row],\
				 feed_dict=input_dict)
	
	return e, u, l


def propagate(sess, w_dict, x, naive_graph):
	tf_x, tf_conv1, tf_max1, tf_conv2, tf_max2, tf_pre_softmax = naive_graph
	x = x.reshape(-1,28,28,1)
	conv1 = sess.run(tf_conv1, feed_dict={tf_x:x})
	max1, mask1, mask_seq1 = np_max(conv1)
	conv2 = sess.run(tf_conv2, feed_dict={tf_max1:max1})
	max2, mask2, mask_seq2 = np_max(conv2)
	o = sess.run(tf_pre_softmax, feed_dict={tf_max2:max2})
	#print o, mask1, mask2
	return o, mask1, mask2, mask_seq1, mask_seq2


def propagate1(sess, w_dict, x, y, tf_graph_param1, ed):
	tf_x1, tf_y1, error_range, estep_decay, upper,\
			lower, param_conv1, param_conv2, param_conv2_input,\
			param_fc, param_fc_input = tf_graph_param1
	#x = x.reshape(-1,28,28,1)
	#o = sess.run(lower, feed_dict={tf_x1:x})
	param = sess.run(param_conv1, feed_dict={tf_x1:x, estep_decay:ed})
	param = np_max2(param[0],param[1],param[2],param[3])

	conv2_dict = {tf_x1:x, estep_decay:ed,\
			param_conv2_input[0]:param[0],\
			param_conv2_input[1]:param[1],\
			param_conv2_input[2]:param[2],\
			param_conv2_input[3]:param[3]}

	param = sess.run(param_conv2, feed_dict=conv2_dict)
	param = np_max2(param[0],param[1],param[2],param[3])

	fc_dict = {tf_x1:x, tf_y1:y, estep_decay:ed,\
			param_fc_input[0]:param[0], param_fc_input[1]:param[1],\
			param_fc_input[2]:param[2], param_fc_input[3]:param[3]}
	param = sess.run(param_fc+[upper,lower], feed_dict=fc_dict)
	return param[1], param[-2], param[-1]





if __name__ == '__main__':
		
	with open('config.json') as config_file:
	  config = json.load(config_file)
	num_eval_examples = config['num_eval_examples']
	eval_batch_size = config['eval_batch_size']
	eval_on_cpu = config['eval_on_cpu']

	model_dir = config['model_dir']

	# Set upd the data, hyperparameters, and the model
	mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

	if eval_on_cpu:
	  with tf.device("/cpu:0"):
		model = Model()
		attack = LinfPGDAttack(model, 
							   config['epsilon'],
							   config['k'],
							   config['a'],
							   config['random_start'],
							   config['loss_func'])
	else:
	  model = Model()
	  attack = LinfPGDAttack(model, 
							 config['epsilon'],
							 config['k'],
							 config['a'],
							 config['random_start'],
							 config['loss_func'])

	global_step = tf.contrib.framework.get_or_create_global_step()

	if not os.path.exists(model_dir):
	  os.makedirs(model_dir)
	eval_dir = os.path.join(model_dir, 'eval')
	if not os.path.exists(eval_dir):
	  os.makedirs(eval_dir)

	last_checkpoint_filename = ''
	already_seen_state = False

	weights = tf.trainable_variables()
	saver = tf.train.Saver()

	filename = tf.train.latest_checkpoint(model_dir)
	with tf.Session() as sess:
		
		saver.restore(sess, filename)
		x = mnist.test.images[:1, :]
		y = mnist.test.labels[:1]

		x_dict = {model.x_input: x,
				  	model.y_input: y}


		y_hat = sess.run(model.pre_softmax, feed_dict=x_dict)
		print np.sum(y_hat), y_hat

		w = sess.run(weights)
		w_dict = {"w_conv1":w[0], "b_conv1":w[1],\
				 "w_conv2":w[2], "b_conv2":w[3],\
				 "w_fc1":w[4], "b_fc1":w[5],\
				 "w_fc2":w[6], "b_fc2":w[7]}
		for wi in w:
			print wi.shape

		naive_graph = build_tf_graph(w_dict)
		tf_graph_param_whole = build_tf_graph_whole(w_dict, 0)

		print propagate_whole(sess, w_dict, x, y,\
			tf_graph_param_whole, naive_graph, 1)



