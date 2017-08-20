from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import os

#TRAIN uses both encoder and decoder, inputX and targets
#ENCODER takes inputX and produces final_state NOTE: Can be feed_forward as well as train
#DECODER takes state and inputX to produce logits NOTE: Add feed_forward in future

#MERGE GRAPHS USING TF VARIABLE AND GET_VARIABLE
#ENCODER WILL OUTPUT FINAL_STATE.
#RUN NN ON FINAL_STATE.
#DECODER WILL GET MID_STATE.
#ENCODER STATE WILL BE NON TRAINABLE NAMED VARIABLE
#NN WILL USE GET_VARIABLE TO GET THE STATE AND RUN THE NN
#DECODER WILL USE GET_VARIABLE TO GET THE NEW MIXED NN STATE AND PROCESS
#TO TRAIN THE TRANSLATOR, CALL DECODER.OUTPUTS AND PROVIDE ENCODER INPUTS

class ModelMode():
	TRAIN = 0
	ENCODE = 1
	DECODE = 2

class DoubleRNNModel():

	def __init__(self, vocab_size, hidden_size=64, learn_rate=0.25, encoding_mode=ModelMode.TRAIN, mid_state=None):

		self.hidden_units = hidden_size
		self.vocab_size = vocab_size
		self.learn_rate = learn_rate
		batch_size = 20

		with tf.variable_scope("RNN") as scope:
			gru_cell = tf.contrib.rnn.GRUCell(self.hidden_units)
			#gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, output_keep_prob=0.75)

		#rev_embed = tf.Variable(tf.truncated_normal([self.hidden_units, vocab_size], stddev=0.01, dtype=tf.float16), name='rev_w')
		rev_embed = tf.get_variable('rev_w', [self.hidden_units, vocab_size], dtype=tf.float16)
		#rev_bias = tf.Variable(tf.truncated_normal([vocab_size], stddev=0.01, dtype=tf.float16), name='rev_b')
		rev_bias = tf.get_variable('rev_b', [vocab_size], dtype=tf.float16)

		if encoding_mode == ModelMode.TRAIN:
			reg_w = tf.get_variable('reg_w', [self.hidden_units, self.hidden_units], dtype=tf.float16)
			reg_b = tf.get_variable('reg_b', [self.hidden_units], dtype=tf.float16)

		with tf.device("/cpu:0"):
			#if encoding_mode == ModelMode.TRAIN or encoding_mode == ModelMode.ENCODE:
			self.inputX = tf.placeholder(tf.int32, shape=[batch_size, None])

			embeddings = tf.get_variable('embedding', [vocab_size, hidden_size], dtype=tf.float16)
			#embeddings = tf.Variable(tf.truncated_normal([vocab_size, hidden_size], stddev=0.01, dtype=tf.float16), name='embedding')

			x = tf.reshape(tf.nn.embedding_lookup(embeddings, self.inputX), [batch_size, -1, self.hidden_units])

			#x = tf.nn.dropout(x, 0.75)
			if encoding_mode == ModelMode.TRAIN:
				self.targets = tf.placeholder(tf.int32, shape=[batch_size, None])
				long_targets = tf.reshape(self.targets, [-1])

			#if encoding_mode == ModelMode.DECODE:
			#	with tf.variable_scope("RNN") as scope:
			#		#self.i_state = tf.placeholder(tf.int32, shape=[self.hidden_units])
			#		self.mid_state = tf.get_variable('mid_state', [1, self.hidden_units], dtype=tf.float16, trainable=False)

		with tf.variable_scope("RNN") as scope:
			if encoding_mode == ModelMode.TRAIN or encoding_mode == ModelMode.ENCODE:
				outputs, self.f_state = tf.nn.dynamic_rnn(gru_cell, x, dtype=tf.float16)

			if encoding_mode == ModelMode.TRAIN:
				scope.reuse_variables()	
				new_state = tf.tanh(tf.matmul(self.f_state, reg_w) + reg_b)
				outputs_2, f_state_2 = tf.nn.dynamic_rnn(gru_cell, x, initial_state=new_state, dtype=tf.float16)

			if encoding_mode == ModelMode.DECODE:
				outputs_2, f_state_2 = tf.nn.dynamic_rnn(gru_cell, x, initial_state=mid_state, dtype=tf.float16)
				self.logits = tf.matmul(tf.reshape(outputs_2, [-1, self.hidden_units]), rev_embed) + rev_bias

			#if encoding_mode == ModelMode.ENCODE or encoding_mode == ModelMode.DECODE:
			#	tvars = tf.trainable_variables()
			#	tvars = []

		if encoding_mode == ModelMode.TRAIN:
			output = tf.reshape(tf.concat(axis=1, values=outputs_2), [-1, self.hidden_units])
			logits =  tf.matmul(output, rev_embed) + rev_bias
			#all_targets = tf.concat([self.targets, self.targets], -1)
			loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [long_targets], [tf.ones_like(long_targets, dtype=tf.float16)])
			self.cost = tf.reduce_sum(loss) / batch_size
			learn_r = tf.Variable(learn_rate, trainable=False)

			tvars = tf.trainable_variables()
			#print(tvars)
			optimizer = tf.train.GradientDescentOptimizer(learn_r)
			gradsvars = optimizer.compute_gradients(self.cost)
			#print(gradsvars)
			grads, _ = tf.clip_by_global_norm([g for g, v in gradsvars], 10)#tf.clip_by_global_norm(tf.gradients(cost, tvars), 10)
			self.train_op = optimizer.apply_gradients(zip(grads, tvars))
			self.new_lr = tf.placeholder(tf.float32, shape=[])
			self.lr_update = tf.assign(learn_r, self.new_lr)

		scope_name = ""
		if encoding_mode == ModelMode.DECODE:
			scope_name = "decode_net"
		else:
			if encoding_mode == ModelMode.ENCODE:
				scope_name = "encode_net"

		all_vars = [v for v in tf.global_variables() if v.name.startswith(scope_name)]
		self.saver = tf.train.Saver(all_vars)

	def run_n_epochs(self, sess, inputX, n_files, n=1):
		avg_err = 0.0
		for e in range(n):
			avg_err = 0.0
			for f in range(n_files):
				cost_eval, _ = sess.run([self.cost, self.train_op], feed_dict={self.inputX: inputX[f][:, :-1], self.targets: inputX[f][:, 1:]})
				avg_err = (avg_err*f + cost_eval)/(f+1)
				if f%150 == 149:
					tf.logging.info(" ".join([str(avg_err), "at epoch", str(f)]))
					sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate/(1+(0.00002*f))})
				#print(avg_err)
			tf.logging.info(" ".join([str(avg_err), "at major epoch", str(e)]))
			self.learn_rate = self.learn_rate*0.95
			sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate})

		return avg_err

class TranslatorModel():
	#FEED FORWARD IMPLEMENTED LATER
	def __init__(self, vocab_size_from, vocab_size_to, hidden_size=64, learn_rate=0.01):

		with tf.variable_scope("encode_net"):
			self.encoder_model = DoubleRNNModel(vocab_size_from, hidden_size=hidden_size, encoding_mode=ModelMode.ENCODE)

		self.vocab_size_to = vocab_size_to
		self.vocab_size_from = vocab_size_from
		self.hidden_size = hidden_size
		self.learn_rate = learn_rate
		batch_size = 20

		self.targets = tf.placeholder(tf.int32, shape=[batch_size, None])
		long_targets = tf.reshape(self.targets, [-1])

		translator_w = tf.get_variable("translator_w", [hidden_size, hidden_size], dtype=tf.float16)
		translator_b = tf.get_variable("translator_b", [hidden_size], dtype=tf.float16)

		"""reg_w = tf.get_variable("reg_w", [hidden_size, hidden_size], dtype=tf.float16)
		reg_b = tf.get_variable("reg_b", [hidden_size], dtype=tf.float16)"""

		#new_state = tf.matmul(tf.tanh(tf.matmul(self.encoder_model.f_state, translator_w) + translator_b), reg_w) + reg_b
		new_state = tf.tanh(tf.matmul(self.encoder_model.f_state, translator_w) + translator_b)

		with tf.variable_scope("decode_net"):
			self.decoder_model = DoubleRNNModel(vocab_size_to, hidden_size=hidden_size, encoding_mode=ModelMode.DECODE, mid_state=new_state)

		logits = self.decoder_model.logits
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [long_targets], [tf.ones_like(long_targets, dtype=tf.float16)])
		self.cost = tf.reduce_sum(loss) / batch_size
		learn_r = tf.Variable(learn_rate, trainable=False)

		tvars = [translator_w, translator_b]#, reg_w, reg_b]#tf.trainable_variables()
		
		optimizer = tf.train.GradientDescentOptimizer(learn_r)
		gradsvars = optimizer.compute_gradients(self.cost, tvars)
		#print(gradsvars)
		grads, _ = tf.clip_by_global_norm([g for g, v in gradsvars], 10)#tf.clip_by_global_norm(tf.gradients(cost, tvars), 10)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))
		self.new_lr = tf.placeholder(tf.float32, shape=[])
		self.lr_update = tf.assign(learn_r, self.new_lr)
		#tf.logging.info([v.name for v in tvars])
		self.saver = tf.train.Saver(tvars)

	def run_n_epochs(self, sess, inputX, inputY, n_files, n=1):
		avg_err = 0.0
		for e in range(n):
			avg_err = 0.0
			for f in range(n_files):
				cost_eval, _ = sess.run([self.cost, self.train_op], feed_dict={self.encoder_model.inputX: inputX[f][:, :-1], self.decoder_model.inputX: inputY[f][:, :-1], self.targets: inputY[f][:, 1:]})
				avg_err = (avg_err*f + cost_eval)/(f+1)
				if f%150 == 149:
					tf.logging.info(" ".join([str(avg_err), "at epoch", str(f)]))
					sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate/(1+(0.00005*f))})
				#print(avg_err)
			tf.logging.info(" ".join([str(avg_err), "at major epoch", str(e)]))
			self.learn_rate = self.learn_rate*0.95
			sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate})

		return avg_err

class HalfTranslatorModel():
	#FEED FORWARD IMPLEMENTED LATER
	def __init__(self, vocab_size_from, vocab_size_to, hidden_size=64, learn_rate=0.01):

		with tf.variable_scope("encode_net"):
			self.encoder_model = DoubleRNNModel(vocab_size_from, hidden_size=hidden_size, encoding_mode=ModelMode.ENCODE)

		self.vocab_size_to = vocab_size_to
		self.vocab_size_from = vocab_size_from
		self.hidden_size = hidden_size
		self.learn_rate = learn_rate
		batch_size = 20

		self.targets = tf.placeholder(tf.int32, shape=[20, None])
		long_targets = tf.reshape(self.targets, [-1])

		translator_w = tf.get_variable("translator_w", [hidden_size, hidden_size], dtype=tf.float16)
		translator_b = tf.get_variable("translator_b", [hidden_size], dtype=tf.float16)

		new_state = tf.tanh(tf.matmul(self.encoder_model.f_state, translator_w) + translator_b)

		with tf.variable_scope("decode_net"):
			self.decoder_model = DoubleRNNModel(vocab_size_to, hidden_size=hidden_size, encoding_mode=ModelMode.DECODE, mid_state=new_state)

		logits = self.decoder_model.logits
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [long_targets], [tf.ones_like(long_targets, dtype=tf.float16)])
		self.cost = tf.reduce_sum(loss) / batch_size
		learn_r = tf.Variable(learn_rate, trainable=False)

		tvars = [translator_w, translator_b]
		tvars += [v for v in tf.trainable_variables() if v.name.startswith("decode_net")]#tf.trainable_variables()
		
		optimizer = tf.train.GradientDescentOptimizer(learn_r)
		gradsvars = optimizer.compute_gradients(self.cost, tvars)
		#print(gradsvars)
		grads, _ = tf.clip_by_global_norm([g for g, v in gradsvars], 0.1)#tf.clip_by_global_norm(tf.gradients(cost, tvars), 10)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))
		self.new_lr = tf.placeholder(tf.float32, shape=[])
		self.lr_update = tf.assign(learn_r, self.new_lr)
		#tf.logging.info([v.name for v in tvars])
		self.saver = tf.train.Saver(tvars)

	def run_n_epochs(self, sess, inputX, inputY, n_files, n=1):
		avg_err = 0.0
		for e in range(n):
			avg_err = 0.0
			for f in range(n_files):
				cost_eval, _ = sess.run([self.cost, self.train_op], feed_dict={self.encoder_model.inputX: inputX[f][:, :-1], self.decoder_model.inputX: inputY[f][:, :-1], self.targets: inputY[f][:, 1:]})
				avg_err = (avg_err*f + cost_eval)/(f+1)
				if f%150 == 149:
					tf.logging.info(" ".join([str(avg_err), "at epoch", str(f)]))
					sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate/(1+(0.00005*f))})
				#print(avg_err)
			tf.logging.info(" ".join([str(avg_err), "at major epoch", str(e)]))
			self.learn_rate = self.learn_rate*0.96
			sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate})

		return avg_err

class TranslatorModel_E2E():
	#FEED FORWARD IMPLEMENTED LATER
	def __init__(self, vocab_size_from, vocab_size_to, hidden_size=64, learn_rate=0.01):

		with tf.variable_scope("encode_net"):
			self.encoder_model = DoubleRNNModel(vocab_size_from, hidden_size=hidden_size, encoding_mode=ModelMode.ENCODE)

		self.vocab_size_to = vocab_size_to
		self.vocab_size_from = vocab_size_from
		self.hidden_size = hidden_size
		self.learn_rate = learn_rate
		batch_size = 20

		self.targets = tf.placeholder(tf.int32, shape=[20, None])
		long_targets = tf.reshape(self.targets, [-1])

		with tf.variable_scope("decode_net"):
			self.decoder_model = DoubleRNNModel(vocab_size_to, hidden_size=hidden_size, encoding_mode=ModelMode.DECODE, mid_state=self.encoder_model.f_state)

		logits = self.decoder_model.logits
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [long_targets], [tf.ones_like(long_targets, dtype=tf.float16)])
		self.cost = tf.reduce_sum(loss) / batch_size
		learn_r = tf.Variable(learn_rate, trainable=False)

		tvars = tf.trainable_variables()
		
		optimizer = tf.train.GradientDescentOptimizer(learn_r)
		gradsvars = optimizer.compute_gradients(self.cost, tvars)
		#print(gradsvars)
		grads, _ = tf.clip_by_global_norm([g for g, v in gradsvars], 10)#tf.clip_by_global_norm(tf.gradients(cost, tvars), 10)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))
		self.new_lr = tf.placeholder(tf.float32, shape=[])
		self.lr_update = tf.assign(learn_r, self.new_lr)
		#tf.logging.info([v.name for v in tvars])
		self.saver = tf.train.Saver(tvars)

	def run_n_epochs(self, sess, inputX, inputY, n_files, n=1):
		avg_err = 0.0
		for e in range(n):
			avg_err = 0.0
			for f in range(n_files):
				cost_eval, _ = sess.run([self.cost, self.train_op], feed_dict={self.encoder_model.inputX: inputX[f][:, :-1], self.decoder_model.inputX: inputY[f][:, :-1], self.targets: inputY[f][:, 1:]})
				avg_err = (avg_err*f + cost_eval)/(f+1)
				if f%150 == 149:
					tf.logging.info(" ".join([str(avg_err), "at epoch", str(f)]))
					sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate/(1+(0.0001*f))})
				#print(avg_err)
			tf.logging.info(" ".join([str(avg_err), "at major epoch", str(e)]))
			self.learn_rate = self.learn_rate*0.95
			sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate})

		return avg_err

class TranslatorModel_E2E_WW():
	#FEED FORWARD IMPLEMENTED LATER
	def __init__(self, vocab_size_from, vocab_size_to, hidden_size=64, learn_rate=0.01):

		with tf.variable_scope("encode_net"):
			self.encoder_model = DoubleRNNModel(vocab_size_from, hidden_size=hidden_size, encoding_mode=ModelMode.ENCODE)

		self.vocab_size_to = vocab_size_to
		self.vocab_size_from = vocab_size_from
		self.hidden_size = hidden_size
		self.learn_rate = learn_rate
		batch_size = 20

		self.targets = tf.placeholder(tf.int32, shape=[batch_size, None])
		long_targets = tf.reshape(self.targets, [-1])

		translator_w = tf.get_variable("translator_w", [hidden_size, hidden_size], dtype=tf.float16)
		translator_b = tf.get_variable("translator_b", [hidden_size], dtype=tf.float16)

		"""reg_w = tf.get_variable("reg_w", [hidden_size, hidden_size], dtype=tf.float16)
		reg_b = tf.get_variable("reg_b", [hidden_size], dtype=tf.float16)"""

		#new_state = tf.matmul(tf.tanh(tf.matmul(self.encoder_model.f_state, translator_w) + translator_b), reg_w) + reg_b
		new_state = tf.tanh(tf.matmul(self.encoder_model.f_state, translator_w) + translator_b)

		with tf.variable_scope("decode_net"):
			self.decoder_model = DoubleRNNModel(vocab_size_to, hidden_size=hidden_size, encoding_mode=ModelMode.DECODE, mid_state=new_state)

		logits = self.decoder_model.logits
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [long_targets], [tf.ones_like(long_targets, dtype=tf.float16)])
		self.cost = tf.reduce_sum(loss) / batch_size
		learn_r = tf.Variable(learn_rate, trainable=False)

		tvars = tf.trainable_variables()
		
		optimizer = tf.train.GradientDescentOptimizer(learn_r)
		gradsvars = optimizer.compute_gradients(self.cost, tvars)
		#print(gradsvars)
		grads, _ = tf.clip_by_global_norm([g for g, v in gradsvars], 10)#tf.clip_by_global_norm(tf.gradients(cost, tvars), 10)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))
		self.new_lr = tf.placeholder(tf.float32, shape=[])
		self.lr_update = tf.assign(learn_r, self.new_lr)
		#tf.logging.info([v.name for v in tvars])
		self.saver = tf.train.Saver(tvars)

	def run_n_epochs(self, sess, inputX, inputY, n_files, n=1):
		avg_err = 0.0
		for e in range(n):
			avg_err = 0.0
			for f in range(n_files):
				cost_eval, _ = sess.run([self.cost, self.train_op], feed_dict={self.encoder_model.inputX: inputX[f][:, :-1], self.decoder_model.inputX: inputY[f][:, :-1], self.targets: inputY[f][:, 1:]})
				avg_err = (avg_err*f + cost_eval)/(f+1)
				if f%150 == 149:
					tf.logging.info(" ".join([str(avg_err), "at epoch", str(f)]))
					sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate/(1+(0.0001*f))})
				#print(avg_err)
			tf.logging.info(" ".join([str(avg_err), "at major epoch", str(e)]))
			self.learn_rate = self.learn_rate*0.95
			sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate})

		return avg_err
