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
	ENCODE = 1
	DECODE = 2

class DoubleRNNModel():

	def __init__(self, vocab_size, hidden_size=64, learn_rate=0.25, encoding_mode=ModelMode.ENCODE, init_state=None):

		self.hidden_units = hidden_size
		self.vocab_size = vocab_size
		self.learn_rate = learn_rate
		batch_size = 20

		with tf.variable_scope("RNN") as scope:
			gru_cell = tf.contrib.rnn.GRUCell(self.hidden_units)

		self.inputX = tf.placeholder(tf.int32, shape=[batch_size, None])

		if encoding_mode == ModelMode.DECODE:
			#rev_embed = tf.Variable(tf.truncated_normal([self.hidden_units, vocab_size], stddev=0.01, dtype=tf.float16), name='rev_w')
			rev_embed = tf.get_variable('rev_w', [self.hidden_units, vocab_size], dtype=tf.float16)
			#rev_bias = tf.Variable(tf.truncated_normal([vocab_size], stddev=0.01, dtype=tf.float16), name='rev_b')
			rev_bias = tf.get_variable('rev_b', [vocab_size], dtype=tf.float16)

		if encoding_mode == ModelMode.ENCODE:
			embeddings = tf.get_variable('embedding', [vocab_size, hidden_size], dtype=tf.float16)
			#embeddings = tf.Variable(tf.truncated_normal([vocab_size, hidden_size], stddev=0.01, dtype=tf.float16), name='embedding')

			x = tf.reshape(tf.nn.embedding_lookup(embeddings, self.inputX), [batch_size, -1, self.hidden_units])


		with tf.variable_scope("RNN") as scope:
			if encoding_mode == ModelMode.ENCODE:
				outputs, self.f_state = tf.nn.dynamic_rnn(gru_cell, x, dtype=tf.float16)


			if encoding_mode == ModelMode.DECODE:
				x = tf.reshape([mid_state for _ in range(self.inputX.get_shape()[1])], [batch_size, -1, self.hidden_units])
				outputs, f_state = tf.nn.dynamic_rnn(gru_cell, x, initial_state=mid_state, dtype=tf.float16)
				self.logits = tf.matmul(tf.reshape(outputs, [-1, self.hidden_units]), rev_embed) + rev_bias

		scope_name = ""
		if encoding_mode == ModelMode.DECODE:
			scope_name = "decode_net"
		else:
			if encoding_mode == ModelMode.ENCODE:
				scope_name = "encode_net"

		all_vars = [v for v in tf.global_variables() if v.name.startswith(scope_name)]
		self.saver = tf.train.Saver(all_vars)

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

		reg_w = tf.get_variable("reg_w", [hidden_size, hidden_size], dtype=tf.float16)
		#reg_b = tf.get_variable("reg_b", [hidden_size], dtype=tf.float16)"""

		self.new_state = tf.matmul(tf.tanh(tf.matmul(self.encoder_model.f_state, translator_w) + translator_b), reg_w)
		#new_state = tf.tanh(tf.matmul(self.encoder_model.f_state, translator_w) + translator_b)

		with tf.variable_scope("decode_net"):
			self.decoder_model = DoubleRNNModel(vocab_size_to, hidden_size=hidden_size, encoding_mode=ModelMode.DECODE, mid_state=self.new_state)

		logits = self.decoder_model.logits
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [long_targets], [tf.ones_like(long_targets, dtype=tf.float16)])
		self.cost = tf.reduce_sum(loss) / batch_size
		learn_r = tf.Variable(learn_rate, trainable=False)

		tvars = [translator_w, translator_b, reg_w]#, reg_b]#tf.trainable_variables()
		
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
				cost_eval, _ = sess.run([self.cost, self.train_op], feed_dict={self.encoder_model.inputX: inputX[f], self.decoder_model.inputX: inputY[f], self.targets: inputY[f]})
				avg_err = (avg_err*f + cost_eval)/(f+1)
				if f%150 == 149:
					tf.logging.info(" ".join([str(avg_err), "at epoch", str(f)]))
				#print(avg_err)
			tf.logging.info(" ".join([str(avg_err), "at major epoch", str(e)]))
			self.learn_rate = self.learn_rate*0.95
			sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate})

		return avg_err
