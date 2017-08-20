from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import os
import random
from train.models import DoubleRNNModel, ModelMode

tf.app.flags.DEFINE_string("model_dir", "rnn2_alphabets", "From model name.")
tf.app.flags.DEFINE_string("model_export", "rnn2_alphabets_en", "From model name.")
#tf.app.flags.DEFINE_string("model_dir_de", "rnn2_phonemes", "To model name.")
#tf.app.flags.DEFINE_string("model_export_de", "rnn2_phonemes_de", "To model name.")
#tf.app.flags.DEFINE_string("to_model", "rnn2_phonemes", "To model name.")
#tf.app.flags.DEFINE_string("to_model_export", "rnn2_phonemes_de", "To model name.")
tf.app.flags.DEFINE_integer("word_size", 64, "Size of each model layer.")
tf.app.flags.DEFINE_integer("vocab_size", 28, "English vocabulary size.")
#tf.app.flags.DEFINE_integer("vocab_size_r", 41, "Phoneme vocabulary size.")
tf.app.flags.DEFINE_integer("mode", 1, "Encode mode")

FLAGS = tf.app.flags.FLAGS

def main(_):
	sess = tf.Session()
	model = None
	scope = ""
	if FLAGS.mode == 1:
		scope = "encode_net"
		model = DoubleRNNModel(FLAGS.vocab_size, hidden_size=FLAGS.word_size)#, encoding_mode=ModelMode.ENCODE)
	else:
		scope = "decode_net"
		model = DoubleRNNModel(FLAGS.vocab_size, hidden_size=FLAGS.word_size)#, encoding_mode=ModelMode.DECODE)

	model.saver.restore(sess, FLAGS.model_dir)
	vars = tf.global_variables()#tf.contrib.framework.list_variables(FLAGS.model_dir_en)
	new_vars = []
	with tf.variable_scope(scope):
		for v in vars:
			nv = None
			#v = tf.contrib.framework.load_variable(FLAGS.model_dir_en, name)
			nv = tf.Variable(v.value(), name=v.name[:-2])
			new_vars.append(nv)#tf.Variable(v.value(), name="/".join([scope, v.name])))

	saver = tf.train.Saver(new_vars)
	sess.run(tf.global_variables_initializer())
	saver.save(sess, FLAGS.model_export)
	"""
	vars = tf.contrib.framework.list_variables(FLAGS.model_dir_de)
	new_vars = []
	for name, shape in vars:
		v = tf.contrib.framework.load_variable(FLAGS.model_dir_de, name)
		new_vars.append(tf.Variable(v, name="/".join(["decode_net", name])))
	saver = tf.train.Saver(new_vars)
	sess.run(tf.global_variables_intilializer())
	saver.save(sess, FLAGS.model_export_de)"""

if __name__ == '__main__':
	tf.app.run()