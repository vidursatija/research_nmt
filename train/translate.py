from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import os
import random
from train.new_models import TranslatorModel #DoubleRNNModel, ModelMode
#from models import DoubleRNNModel, ModelMode

#tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_float("learn_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("word_size", 64, "Size of each model layer.")
tf.app.flags.DEFINE_integer("vocab_size", 28, "English vocabulary size.")
tf.app.flags.DEFINE_integer("vocab_size_r", 41, "Phoneme vocabulary size.")
tf.app.flags.DEFINE_string("pickle_dir", "words_phonemes.p", "Data directory")
tf.app.flags.DEFINE_string("model_dir", ".", "Training directory.")
tf.app.flags.DEFINE_string("model_name", "alpha_to_phoneme_pretrain", "Model name.")
tf.app.flags.DEFINE_string("from_model", "rnn2_alphabets", "From model name.")
tf.app.flags.DEFINE_string("to_model", "rnn2_phonemes", "To model name.")
#tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1, "How many training steps to do per checkpoint.")
#tf.app.flags.DEFINE_boolean("feed_forward", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_integer("n_steps", 100, "Number of major epochs to run")
#tf.app.flags.DEFINE_integer("gcloud", False, "Enable if running on cloud")

FLAGS = tf.app.flags.FLAGS

def train():
	sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.75)))

	pickle_file = tf.read_file(FLAGS.pickle_dir)
	file_data = pickle.loads(sess.run(pickle_file))
	all_alphabets = file_data["input"][0]
	all_phonemes = file_data["input"][0]

	m = TranslatorModel(FLAGS.vocab_size, FLAGS.vocab_size_r, hidden_size=FLAGS.word_size, learn_rate=FLAGS.learn_rate)
	ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
	sess.run(tf.global_variables_initializer())
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		tf.logging.info("Imported model")
		m.saver.restore(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))
		#RESTORE TO AND FROM MODELS
		with tf.variable_scope("encode_net"):
			tf.logging.info("Model encoder")
			m.encoder_model.saver.restore(sess, os.path.join(FLAGS.model_dir, "encoder"))
		with tf.variable_scope("decode_net"):
			tf.logging.info("Model decoder")
			m.decoder_model.saver.restore(sess, os.path.join(FLAGS.model_dir, "decoder"))

	train_writer = tf.summary.FileWriter(FLAGS.model_dir)

	for i in range(int(FLAGS.n_steps)):
		err = m.run_n_epochs(sess, train_writer, all_alphabets, all_phonemes, len(all_alphabets), n=1, epoch_num=i)
		if i%10 == 9:
			tf.logging.info("Saving model")
			m.saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))
			m.encoder_model.saver.save(sess, os.path.join(FLAGS.model_dir, "encoder"))
			m.decoder_model.saver.save(sess, os.path.join(FLAGS.model_dir, "decoder"))
	train_writer.close()
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
