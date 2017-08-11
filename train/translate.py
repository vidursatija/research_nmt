from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
#from model import Model

tf.logging.set_verbosity(tf.logging.INFO)

class DoubleRNNModel():

	def __init__(self, vocab_size, feed_forward=False, hidden_size=64, learn_rate=0.25):

		self.hidden_units = hidden_size
		self.vocab_size = vocab_size
		self.learn_rate = learn_rate

		with tf.variable_scope("RNN") as scope:
			gru_cell = tf.contrib.rnn.GRUCell(self.hidden_units)

			#if feed_forward == False:
			#	gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, output_keep_prob=0.75)#tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size), output_keep_prob=0.85), tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size), output_keep_prob=0.85)])#tf.contrib.rnn.GRUCell(hidden_size)

		rev_embed = tf.get_variable('rev_w', [self.hidden_units, vocab_size], dtype=tf.float16)
		#rev_embed_r = tf.Variable(tf.truncated_normal([hidden_size_r, vocab_size_r], stddev=0.01, dtype=tf.float16), name='rev_w_r')
		rev_bias = tf.get_variable('rev_b', [vocab_size], dtype=tf.float16)

		#neo_cortex_w1 = tf.get_variable("neo_weights1", [self.hidden_units, self.hidden_units], dtype=tf.float16)
		#neo_cortex_b1 = tf.get_variable("neo_bias1", [self.hidden_units], dtype=tf.float16)

		with tf.device("/cpu:0"):
			self.inputX = tf.placeholder(tf.int32, shape=[None])

			embeddings = tf.get_variable('embedding', [vocab_size, hidden_size], dtype=tf.float16)

			x = tf.reshape(tf.nn.embedding_lookup(embeddings, self.inputX), [1, -1, self.hidden_units])

			if feed_forward == False:
				#x = tf.nn.dropout(x, 0.75)
				self.targets = tf.placeholder(tf.int32, shape=[None])

		with tf.variable_scope("RNN") as scope:
			outputs, f_state = tf.nn.dynamic_rnn(gru_cell, x, dtype=tf.float16)
			scope.reuse_variables()
			#	convert_state = tf.tanh(tf.matmul(f_state, neo_cortex_w1)+neo_cortex_b1)
			if feed_forward == False:
				outputs_2, f_state_2 = tf.nn.dynamic_rnn(gru_cell, x, initial_state=f_state, dtype=tf.float16)

		#Add tf while loop to control feed forward
		#vocab_size-2 is finish in word to id
		if feed_forward:
			"""self.feed_forward_outputs = [tf.argmax(tf.matmul(outputs[0][-1], rev_embed) + rev_bias, axis=0)]

			def body(state):
				#convert ffo to [1, 1, self.hidden_units]
				iX = tf.reshape(self.feed_forward_outputs[-1], [1, 1]) #number
				iX_r = tf.reshape(wi2pi[self.feed_forward_outputs[-1]], [1, 1])
				whole_x_i = tf.concat([tf.nn_embedding_lookup(embeddings, iX), tf.nn.embedding_lookup(embeddings_r, iX_r)], 2)
				op, new_state = lstm_cell(whole_x_i, state)
				y_0 = tf.argmax(tf.matmul(op[0], rev_embed) + rev_bias, axis=0)
				self.feed_forward_outputs.append(y_0)
				return new_state

			def condition(state):
				return tf.not_equal(feed_forward_outputs[-1], vocab_size-2)

			with tf.variable_scope("RNN"):
				final_state = tf.while_loop(condition, body, loop_vars=[f_state], shape_invariants=[f_state.get_shape()])"""
			pass

		else:
			output = tf.reshape(tf.concat(axis=1, values=outputs_2), [-1, self.hidden_units])
			logits =  tf.matmul(output, rev_embed) + rev_bias
			#all_targets = tf.concat([self.targets, self.targets], -1)
			loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [self.targets], [tf.ones_like(self.targets, dtype=tf.float16)])
			self.cost = tf.reduce_sum(loss)
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
		print(tf.global_variables())
		self.saver = tf.train.Saver(tf.global_variables())

	def run_n_epochs(self, sess, inputX, n_files, n=1):
		avg_err = 0.0
		for e in range(n):
			avg_err = 0.0
			for f in range(n_files):
				cost_eval, _ = sess.run([self.cost, self.train_op], feed_dict={self.inputX: inputX[f][:-1], self.targets: inputX[f][1:]})
				avg_err = (avg_err*f + cost_eval)/(f+1)
				if f%1500 == 1499:
					tf.logging.info(" ".join([str(avg_err), "at epoch", str(f)]))
					sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate/(1+(0.00003*f))})
				#print(avg_err)
			tf.logging.info(" ".join([str(avg_err), "at major epoch", str(e)]))
			self.learn_rate = self.learn_rate*0.8
			sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate})

		return avg_err

	def run_prediction(self, sess, start_lyrics, start_rhymes):
		f_outs = sess.run(self.feed_forward_outputs, feed_dict={self.inputX: start_lyrics, self.inputX_r: start_rhymes})
		return f_outs

tf.app.flags.DEFINE_float("learn_rate", 0.15, "Learning rate.")
tf.app.flags.DEFINE_integer("word_size", 64, "Size of each model layer.")
tf.app.flags.DEFINE_integer("vocab_size", 28, "English vocabulary size.")
tf.app.flags.DEFINE_integer("vocab_size_r", 28, "English vocabulary size.")
tf.app.flags.DEFINE_string("pickle_dir", "words_phonemes.p", "Data directory")
tf.app.flags.DEFINE_string("model_dir", "rnn2_alphabets", "Training directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 2, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("feed_forward", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_integer("n_steps", 50000, "Number of major epochs to run")

FLAGS = tf.app.flags.FLAGS

def train():
	sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))

	pickle_file = tf.read_file(FLAGS.pickle_dir)
	file_data = pickle.loads(sess.run(pickle_file))
	all_alphabets = file_data["input"][0]

	m = DoubleRNNModel(FLAGS.vocab_size, feed_forward=FLAGS.feed_forward, hidden_size=FLAGS.word_size, learn_rate=FLAGS.learn_rate)
	
	sess.run(tf.global_variables_initializer())

	for i in range(int(FLAGS.n_steps/FLAGS.steps_per_checkpoint)):
		err = m.run_n_epochs(sess, all_alphabets, len(all_alphabets), n=int(FLAGS.steps_per_checkpoint))
		#if i%2 == 1:
		m.saver.save(sess, FLAGS.model_dir)
		#m.learn_rate = m.learn_rate*0.95
		#tf.logging.info(" ".join([str(err), "at major epoch", str(i)]))
		#sess.run(tf.Print(err, [i, err]))

def predict():
	sess = tf.Session()#config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))

	pickle_file = tf.read_file(FLAGS.pickle_dir)
	file_data = pickle.loads(sess.run(pickle_file))
	
	m = DoubleRNNModel(FLAGS.vocab_size, feed_forward=FLAGS.feed_forward, hidden_size=FLAGS.word_size, learn_rate=FLAGS.learn_rate)
	m.saver.restore(sess, FLAGS.model_dir)


def main(_):
	if FLAGS.feed_forward == False:
		train()
	else:
		predict()

if __name__ == '__main__':
	tf.app.run()

		

