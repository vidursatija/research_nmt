from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import os
import random
from train.models import DoubleRNNModel, ModelMode
#from models import DoubleRNNModel, ModelMode

#tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_float("learn_rate", 0.15, "Learning rate.")
tf.app.flags.DEFINE_integer("word_size", 64, "Size of each model layer.")
tf.app.flags.DEFINE_integer("vocab_size", 28, "English vocabulary size.")
tf.app.flags.DEFINE_string("pickle_dir", "words_phonemes.p", "Data directory")
tf.app.flags.DEFINE_string("model_dir", "rnn2_alphabets", "Training directory.")
tf.app.flags.DEFINE_string("model_name", "rnn2_alphabets", "Model name.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 2, "How many training steps to do per checkpoint.")
#tf.app.flags.DEFINE_boolean("feed_forward", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_integer("n_steps", 50000, "Number of major epochs to run")
#tf.app.flags.DEFINE_integer("gcloud", False, "Enable if running on cloud")

FLAGS = tf.app.flags.FLAGS

def train():
	sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))

	pickle_file = tf.read_file(FLAGS.pickle_dir)
	file_data = pickle.loads(sess.run(pickle_file))
	all_alphabets = file_data["input"][0]
	#all_alphabets = file_data["input"][1]

	m = DoubleRNNModel(FLAGS.vocab_size, hidden_size=FLAGS.word_size, learn_rate=FLAGS.learn_rate, encoding_mode=ModelMode.TRAIN)
	ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		tf.logging.info("Imported model")
		m.saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		sess.run(tf.global_variables_initializer())

	for i in range(int(FLAGS.n_steps/FLAGS.steps_per_checkpoint)):
		err = m.run_n_epochs(sess, all_alphabets, len(all_alphabets), n=int(FLAGS.steps_per_checkpoint))
		#if i%2 == 1:
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			m.saver.save(sess, ckpt.model_checkpoint_path)
		else:
			m.saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))
		#m.learn_rate = m.learn_rate*0.95
		#tf.logging.info(" ".join([str(err), "at major epoch", str(i)]))
		#sess.run(tf.Print(err, [i, err]))

"""def predict():
	sess = tf.Session()#config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))

	pickle_file = tf.read_file(FLAGS.pickle_dir)
	file_data = pickle.loads(sess.run(pickle_file))
	#letter_to_index = file_data["lti"]
	all_alphabets = file_data["input"][0]
	#phoneme_to_index = file_data["pti"]
	
	m = DoubleRNNModel(FLAGS.vocab_size, hidden_size=FLAGS.word_size, learn_rate=FLAGS.learn_rate)
	ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		tf.logging.info("Imported model")
		m.saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		tf.logging.info("No model found")
		return

	for i in range(10):
		z = random.randint(0, len(all_alphabets)-1)
		per = m.run_prediction(sess, all_alphabets[z])
		tf.logging.info(" ".join(["Word len:", str(len(all_alphabets[z])), "per:", str(per)]))"""

def main(_):
	#if FLAGS.feed_forward == False:
	train()
	#else:
	#	predict()

if __name__ == '__main__':
	tf.app.run()
