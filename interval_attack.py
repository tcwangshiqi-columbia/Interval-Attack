import tensorflow as tf
import numpy as np
import json
import sys
import math
from tensorflow.examples.tutorials.mnist import input_data
from scipy.spatial.distance import pdist, squareform
from model import Model
import argparse
import time

from symbolic_interval import *


def cw_attack(sess, model, x, y, config, x_nat=None):
	if(x_nat is None):
		x_nat = x
	x_adv = np.copy(x)
	y_adv = np.copy(y)

	for i in range(args.iters):
		g, p = sess.run([model.cw_grad, model.y_pred], feed_dict={
						model.x_input: x_adv,
						model.y_input: y_adv
					})


		x_adv += config['a'] * np.sign(g)

		x_adv = np.clip(x_adv, x_nat - config['epsilon'], x_nat + config['epsilon']) 
		x_adv = np.clip(x_adv, 0, 1)
		
	return x_adv, y_adv


def pgd_attack(sess, model, x, y, config, x_nat=None):
	if(x_nat is None):
		x_nat = x
	x_adv = np.copy(x)
	y_adv = np.copy(y)

	for i in range(args.iters):
		g, p = sess.run([model.pgd_grad, model.y_pred], feed_dict={
						model.x_input: x_adv,
						model.y_input: y_adv
					})

		x_adv += config['a'] * np.sign(g)

		x_adv = np.clip(x_adv, x_nat - config['epsilon'], x_nat + config['epsilon']) 
		x_adv = np.clip(x_adv, 0, 1) 
		
	return x_adv, y_adv

def svgd_kernel(theta):
	sq_dist = pdist(theta)
	pairwise_dists = squareform(sq_dist)**2

	h = np.median(pairwise_dists)  
	h = np.sqrt(0.5 * h / np.log(theta.shape[0]))

	# compute the rbf kernel
	Kxy = np.exp( -pairwise_dists / h**2 / 2)

	dxkxy = -np.matmul(Kxy, theta)
	sumkxy = np.sum(Kxy, axis=1)
	for i in range(theta.shape[1]):
	  dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)
	dxkxy = dxkxy / (h**2)
	return (Kxy, dxkxy)




def wgf_kernel(theta):
	sq_dist = pdist(theta)
	pairwise_dists = squareform(sq_dist)**2
	
	h = np.median(pairwise_dists)  
	h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

	Kxy = np.exp( -pairwise_dists / h**2 / 2)
	Kxy = np.multiply((pairwise_dists / h**2 / 2 - 1), Kxy)
	
	dxkxy = -np.matmul(Kxy, theta)
	sumkxy = np.sum(Kxy, axis=1)
	for i in range(theta.shape[1]):
	  dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)

	return (Kxy, dxkxy)


def blob(sess, model, x, y, config, x_nat):
	"""Given a set of examples (x_nat, y), returns a set of adversarial
	   examples within epsilon of x_nat in l_infinity norm."""
	
	x_adv = np.copy(x)
	batch_size = x_adv.shape[0]  

	idx = np.arange(x.shape[0])
	permutation = np.arange(x.shape[0])
	
	for epoch in range(20):
		np.random.shuffle(idx)
		x_nat, x_adv, y = x_nat[idx], x_adv[idx], y[idx]
		permutation = permutation[idx]

		for ei in range(10):

			tf_grad = model.blob_grad
			grad = sess.run(tf_grad, feed_dict={model.x_input: x_adv,
											   model.y_input: y})
											 
										
			kxy, dxkxy = svgd_kernel(x_adv)
			x_adv += config['a'] * np.sign(1.1*(-(np.matmul(kxy, -grad) + dxkxy)/batch_size) + grad)

			x_adv = np.clip(x_adv, x_nat - config['epsilon'], x_nat + config['epsilon']) 
			x_adv = np.clip(x_adv, 0, 1) # ensure valid pixel range

	inv_permutation = np.argsort(permutation)
	x_adv = x_adv[inv_permutation]
	x_nat = x_nat[inv_permutation]
	y = y[inv_permutation]

	return x_adv


def dgf(sess, model, x, y, config, x_nat):
	"""Given a set of examples (x_nat, y), returns a set of adversarial
	   examples within epsilon of x_nat in l_infinity norm."""
	
	x_adv = np.copy(x)
	batch_size = x_adv.shape[0] 

	idx = np.arange(x.shape[0])
	permutation = np.arange(x.shape[0])

	for epoch in range(20):
		np.random.shuffle(idx)
		x_nat, x_adv, y = x_nat[idx], x_adv[idx], y[idx]
		permutation = permutation[idx]
		
		for ei in range(5):

			tf_grad = model.blob_grad
			grad = sess.run(tf_grad, feed_dict={model.x_input: x_adv,
											   model.y_input: y})
											 
			c = -0.001																 
										
			kxy, dxkxy = wgf_kernel(x_adv)
			x_adv += config['a'] * np.sign(c*dxkxy + grad)


			x_adv = np.clip(x_adv, x_nat - config['epsilon'], x_nat + config['epsilon']) 
			x_adv = np.clip(x_adv, 0, 1) # ensure valid pixel range

	inv_permutation = np.argsort(permutation)
	x_adv = x_adv[inv_permutation]
	x_nat = x_nat[inv_permutation]
	y = y[inv_permutation]

	return x_adv

	
	
def eval(x, y, sess, model, blist = None):
	y_p = sess.run(model.y_pred,\
				feed_dict={model.x_input:x,\
				model.y_input:y
			})
	if(blist is not None):
		#print blist
		blist = np.logical_or(blist, np.not_equal(y_p, y))
		acc = np.sum((1-blist).astype(np.float32))
	else:
		acc = np.sum(np.equal(y_p, y).astype(np.float32))
	acc = acc / float(y.shape[0]) * 100.0
	return acc




def interval_attack_exchange1(sess, model,\
					tf_graph_param_whole, naive_graph, x, x_adv, y, config, bool_list):
	#x_adv = np.copy(x)
	y_adv = np.copy(y)
	start = time.time()

	for i in range(args.interval_iters):
		
		if(args.im==INTERVAL_CW):
			x_adv, y_adv = cw_attack(sess, model, x_adv, y_adv, config, x)
		if(args.im==INTERVAL_PGD):
			x_adv, y_adv = pgd_attack(sess, model, x_adv, y_adv, config, x)
		if(args.im==INTERVAL_DGF):
			x_adv_new = dgf(sess, model, x_adv, y, config, x_nat=x)
			x_adv = x_adv_new*(1.0-bool_list).reshape(-1,1)+x_adv*bool_list.reshape(-1,1)
		if(INTERVAL_BLOB):
			x_adv_new = blob(sess, model, x_adv, y, config, x_nat=x)
			x_adv = x_adv_new*(1.0-bool_list).reshape(-1,1)+x_adv*bool_list.reshape(-1,1)
			 

		p = sess.run(model.y_pred, feed_dict={
						model.x_input: x_adv,
						model.y_input: y_adv
					})
		bool_list = np.logical_or(bool_list, np.not_equal(p, y))
		print "iter:", i, " adv_acc:", (1.0-np.sum(bool_list.astype(np.float32))/x.shape[0])*100.0

		for j in range(x.shape[0]):
			if(bool_list[j]):
				continue

			ed = 1
			stuck = False
			while(True):
				ed = min(ed, config['epsilon']/2/args.estep)
				eq, u, l = propagate_whole(sess, w_dict, x_adv[j:j+1], y[j:j+1], tf_graph_param_whole, naive_graph, ed)
				'''
				eq, u, l = sess.run([equation,
										upper, lower], feed_dict={
										model.x_input:x_adv[j:j+1],
										model.y_input:y_adv[j:j+1],
										estep_decay:ed
									})
				'''
				d = np.argmax(u)
				#print "iter:", i, "sample:", j, "estep_decay:", ed, "new_estep:", args.estep*ed, "robust_output:", d, "true_label:", y[j]
				#print "upper:", u

				if(d == y_adv[j]):
					# print "Still the same label for the range"
					if(ed == config['epsilon']/2/args.estep):
						stuck = True
						break
					ed *= 3
				else:
					# print "label changed, break"
					eq = eq[0, :, d]
					break

			if(stuck): continue

			pos = (eq>0).astype(np.float32)*2*ed
			pos = pos - ed

			x_adv[j] += pos

		x_adv = np.clip(x_adv, x - config['epsilon'], x + config['epsilon']) 
		x_adv = np.clip(x_adv, 0, 1)

	if(args.im==INTERVAL_CW):
		x_adv, y_adv = cw_attack(sess, model, x_adv, y_adv, config, x)
	if(args.im==INTERVAL_PGD):
		x_adv, y_adv = pgd_attack(sess, model, x_adv, y_adv, config, x)
	if(args.im==INTERVAL_DGF):
		x_adv_new = dgf(sess, model, x_adv, y_adv, config, x_nat=x)
		x_adv = x_adv_new*(1.0-bool_list).reshape(-1,1)+x_adv*bool_list.reshape(-1,1)
	if(INTERVAL_BLOB):
		x_adv_new = blob(sess, model, x_adv, y, config, x_nat=x)
		x_adv = x_adv_new*(1.0-bool_list).reshape(-1,1)+x_adv*bool_list.reshape(-1,1)

	p = sess.run(model.y_pred, feed_dict={
					model.x_input: x_adv,
					model.y_input: y_adv
				})
	bool_list = np.logical_or(bool_list, np.not_equal(p, y))

	return x_adv, y_adv, bool_list



def check(x, x_nat, epsilon):
	for i in range(x.shape[0]):
		#print (x_nat-x)
		for xi, xi_nat in zip(x[i], x_nat[i]):
			if((xi>=xi_nat+epsilon+0.00001) or (xi<=xi_nat-epsilon-0.00001)):
				print "wrong adv generated!"
				print xi-xi_nat
				exit(1)
			if(xi>1 or xi<0):
				print "wrong adv generated, out of 0 and 1!"
				print xi
				exit(1)



# method
PGD = 1
CW = 2
INTERVALGD = 3
VERIFIED = 4
WHITE = 5
INTERVAL_FREQUENT = 6
INTERVAL_SPSA = 7
INTERVAL_WIDTH = 8
INTERVAL_EX = 9
INTERVAL_EX1 = 10
BLOB = 11
DGF = 12

INTERVAL_ITER = 40

#interval grad method
INTERVAL_BASE = 0
INTERVAL_PGD = 1
INTERVAL_CW = 2
INTERVAL_DGF = 3
INTERVAL_BLOB = 4

np.random.seed(0)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--method', type=int, default=INTERVAL_EX1)
	parser.add_argument('--baseline', action='store_true', default=False)
	parser.add_argument('--im', type=int, default=INTERVAL_BLOB)
	parser.add_argument('--random', type=int, default=50)
	# epsilon used for estimating the interval gradient
	parser.add_argument('--estep', type=float, default=0.01)
	# batch_size that is used for calculating symbolic propagation
	# in case out of memory
	parser.add_argument('--batch_size', type=int, default=500)
	# how many batches needed to test
	parser.add_argument('--batches', type=int, default=1)
	parser.add_argument('--iters', type=int, default=40)
	parser.add_argument('--interval_iters', type=int, default=3)
	parser.add_argument('--estep_decay', type=float, default=0.94)
	parser.add_argument('--check', action='store_true', default=True)
	args = parser.parse_args()


	with open('config.json') as config_file:
	  config = json.load(config_file)
	num_eval_examples = config['num_eval_examples']
	eval_batch_size = config['eval_batch_size']
	eval_on_cpu = config['eval_on_cpu']

	model_dir = config['model_dir']

	# Set upd the data, hyperparameters, and the model
	mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
	x = mnist.test.images[:args.batch_size*args.batches]
	y = mnist.test.labels[:args.batch_size*args.batches]


	if eval_on_cpu:
	  with tf.device("/cpu:0"):
		model = Model()
	else:
	  model = Model()

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

	model_file = tf.train.latest_checkpoint(model_dir)
	

	if(args.method==INTERVAL_EX1):
		with tf.Session() as sess:
			saver.restore(sess, model_file)

			w = sess.run(weights)
			w_dict = {"w_conv1":w[0], "b_conv1":w[1],\
					 "w_conv2":w[2], "b_conv2":w[3],\
					 "w_fc1":w[4], "b_fc1":w[5],\
					 "w_fc2":w[6], "b_fc2":w[7]}
		# tf_x1, tf_y1, error_range, estep_decay, upper, lower, param_conv1, param_conv2, param_conv2_input, param_fc, param_fc_input
		naive_graph = build_tf_graph(w_dict)
		tf_graph_param_whole = build_tf_graph_whole(w_dict, args.estep)

		d_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=model.y_input, logits=tf_graph_param_whole[7])

		interval_loss = d_xent

	start = time.time()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, model_file)

		x_adv_list = []

		for batch in range(args.batches):

			start_batch = batch*args.batch_size
			end_batch = start_batch + args.batch_size
			print "batch", batch, "start:", start_batch, "end:", end_batch

			x = mnist.test.images[start_batch:end_batch]
			y = mnist.test.labels[start_batch:end_batch]

			if(args.method==PGD):
				
				x_adv = np.copy(x)
				bool_list = np.zeros(x.shape[0]).astype(np.bool)
				for rs in range(args.random):
					print "random starts:", rs
					#x_adv = x_adv + np.random.uniform(-config['epsilon'], config['epsilon'], x.shape)
					x_adv = (x + np.random.uniform(-config['epsilon'], config['epsilon'], x.shape))*(1.0-bool_list).reshape(-1,1)+x_adv*bool_list.reshape(-1,1)
					x_adv = np.clip(x_adv, 0, 1)
					x_adv, y_adv = pgd_attack(sess, model, x_adv, y, config, x_nat=x)

					p = sess.run(model.y_pred, feed_dict={
							model.x_input: x_adv,
							model.y_input: y
						})
					bool_list = np.logical_or(bool_list, np.not_equal(p, y))
					print "adv_acc:", (1.0-np.sum(bool_list.astype(np.float32))/x.shape[0])*100.0

				acc = eval(x, y, sess, model)
				adv_acc = eval(x_adv, y, sess, model)

				print "adv_acc:", adv_acc, " acc:", acc

				

			elif(args.method==CW):

				x_adv = np.copy(x)
				bool_list = np.zeros(x.shape[0]).astype(np.bool)
				for rs in range(args.random):
					print "random starts:", rs
					#x_adv = x_adv + np.random.uniform(-config['epsilon'], config['epsilon'], x.shape)
					x_adv = (x + np.random.uniform(-config['epsilon'], config['epsilon'], x.shape))*(1.0-bool_list).reshape(-1,1)+x_adv*bool_list.reshape(-1,1)
					x_adv = np.clip(x_adv, 0, 1)
					x_adv, y_adv = cw_attack(sess, model, x_adv, y, config, x_nat=x)

					p = sess.run(model.y_pred, feed_dict={
							model.x_input: x_adv,
							model.y_input: y
						})
					bool_list = np.logical_or(bool_list, np.not_equal(p, y))
					print "adv_acc:", (1.0-np.sum(bool_list.astype(np.float32))/x.shape[0])*100.0

				acc = eval(x, y, sess, model)
				adv_acc = eval(x_adv, y, sess, model)

				print "adv_acc:", adv_acc, " acc:", acc


			elif(args.method==VERIFIED):
				pass

			elif(args.method==BLOB):
				x_adv = np.copy(x)
				bool_list = np.zeros(x.shape[0]).astype(np.bool)
				for rs in range(args.random):
					print "random starts:", rs
					#x_adv = x_adv + np.random.uniform(-config['epsilon'], config['epsilon'], x.shape)
					x_adv = (x + np.random.uniform(-config['epsilon'], config['epsilon'], x.shape))*(1.0-bool_list).reshape(-1,1)+x_adv*bool_list.reshape(-1,1)
					x_adv = np.clip(x_adv, 0, 1)
					x_adv_new = blob(sess, model, x_adv, y, config, x_nat=x)
					x_adv = x_adv_new*(1.0-bool_list).reshape(-1,1)+x_adv*bool_list.reshape(-1,1)

					p = sess.run(model.y_pred, feed_dict={
							model.x_input: x_adv,
							model.y_input: y
						})
					bool_list = np.logical_or(bool_list, np.not_equal(p, y))
					print "adv_acc:", (1.0-np.sum(bool_list.astype(np.float32))/x.shape[0])*100.0

				acc = eval(x, y, sess, model)
				adv_acc = eval(x_adv, y, sess, model)

				print "adv_acc:", adv_acc, " acc:", acc

			elif(args.method==DGF):
				x_adv = np.copy(x)
				bool_list = np.zeros(x.shape[0]).astype(np.bool)
				for rs in range(args.random):
					print "random starts:", rs
					#x_adv = x_adv + np.random.uniform(-config['epsilon'], config['epsilon'], x.shape)
					x_adv = (x + np.random.uniform(-config['epsilon'], config['epsilon'], x.shape))*(1.0-bool_list).reshape(-1,1)+x_adv*bool_list.reshape(-1,1)
					x_adv = np.clip(x_adv, 0, 1)

					x_adv = dgf(sess, model, x_adv, y, config, x_nat=x)

					p = sess.run(model.y_pred, feed_dict={
							model.x_input: x_adv,
							model.y_input: y
						})
					bool_list = np.logical_or(bool_list, np.not_equal(p, y))
					print "adv_acc:", (1.0-np.sum(bool_list.astype(np.float32))/x.shape[0])*100.0

				acc = eval(x, y, sess, model)
				adv_acc = eval(x_adv, y, sess, model)

				print "adv_acc:", adv_acc, " acc:", acc


			elif(args.method==INTERVAL_EX1):

				x_adv = []
				y_adv = []
				blist = []
					
				x_adv = np.copy(x)
				bool_list = np.zeros(x.shape[0]).astype(np.bool)

				for rs in range(args.random):
					if(rs>=2): args.interval_iters = 1
					print "random starts:", rs
					if(rs!=0):
						x_adv = (x + np.random.uniform(-config['epsilon'], config['epsilon'], x.shape))*(1.0-bool_list).reshape(-1,1)+x_adv*bool_list.reshape(-1,1)
						x_adv = np.clip(x_adv, 0, 1)
					x_adv, y_adv, bool_list = interval_attack_exchange1(sess, model,\
										tf_graph_param_whole, naive_graph,\
										x, x_adv, y, config, bool_list)

					#blist.append(bool_list)
					#print np.sum(bool_list.astype(np.float32))/x1.shape[0]*100

					acc = eval(x, y, sess, model)
					adv_acc = eval(x_adv, y, sess, model)

					print "adv_acc:", adv_acc, " acc:", acc
				
			else:
				x_adv = []
				y_adv = []
				for i in range(args.batches):
					
					x1 = x[i*args.batch_size:(i+1)*args.batch_size]
					y1 = y[i*args.batch_size:(i+1)*args.batch_size]
					
					x1, y1 = interval_attack(sess, model, interval_grad,\
									estep_decay, x1, y1, config)
					x_adv.append(x1)
					y_adv.append(y1)
				
				x_adv = np.array(x_adv).reshape(-1, 784)
				y_adv = np.array(y_adv).reshape(-1)

				if(args.im==INTERVAL_CW):
					x_adv, y_adv = cw_attack(sess, model, x_adv, y_adv, config, x)
				if(args.im==INTERVAL_PGD):
					x_adv, y_adv = pgd_attack(sess, model, x_adv, y_adv, config, x)

				acc = eval(x, y, sess, model)
				adv_acc = eval(x_adv, y_adv, sess, model)

				if(args.check):
					check(x_adv, x, config['epsilon'])

				print "adv_acc:", adv_acc, " acc:", acc

			if(args.check):
				x_nat = mnist.test.images[start_batch:end_batch]
				check(x_adv, x_nat, config['epsilon'])
				print ("check passed!")

			y_nat = mnist.test.labels[start_batch:end_batch]
			adv_acc = eval(x_adv, y_nat, sess, model)
			print "final adv_acc:", adv_acc
			np.save("x_adv_attack1_"+str(args.method)+"_b"+str(batch), x_adv)
			x_adv_list.append(x_adv) 

			print "time:", time.time()-start
		


		x_adv = np.array(x_adv_list).reshape(-1,784)
		if(args.check):
			x_nat = mnist.test.images
			check(x_adv, x_nat, config['epsilon'])
			print ("Final check passed!")

		y_nat = mnist.test.labels
		adv_acc = eval(x_adv, y_nat, sess, model)
		print "Final adv_acc:", adv_acc
		np.save("x_adv_attack1_"+str(args.method)+"_final", x_adv)

		print "time:", time.time()-start