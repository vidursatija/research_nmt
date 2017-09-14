import nltk
import re
import numpy as np
import pickle
from collections import Counter

class BatchGen():
	def __init__(self):
		self.arpabet = nltk.corpus.cmudict.dict()
		count_dict = dict()
		len_pairs = []

		for key, value in self.arpabet.items():
			c_word = re.sub('[^a-zA-Z]', '', key)
			c_word = c_word.lower()
			len_pairs.append(str((len(c_word), len(value[0]))))
			for p in value:
				for phoneme in p:
					if phoneme[-1] == '0' or phoneme[-1] == '1' or phoneme[-1] == '2':
						phoneme = phoneme[:-1]
					try:
						count_dict[str(phoneme)] += 1
					except:
						count_dict[str(phoneme)] = 1

		batch_len = Counter(len_pairs)
		batch_to_id = dict()
		major_batch_count = 0
		for key, value in batch_len.items():
			if value < 10:
				continue
			try:
				if batch_to_id[key] >= 0:
					pass
				else:
					batch_to_id[key] = major_batch_count
					major_batch_count += 1
			except Exception as e:
				batch_to_id[key] = major_batch_count
				major_batch_count += 1
		#print(batch_len)
		#print(batch_to_id)

		self.letter_to_index = dict()
		self.index_to_letter = list()
		#self.letter_to_index['<start>'] = 0
		#self.index_to_letter.append('<start>')
		self.letter_to_index['<end>'] = 0
		self.index_to_letter.append('<end>')
		for index in range(0, 26):
			self.letter_to_index[chr(index+97)] = index+1
			self.index_to_letter.append(chr(index+97))

		self.phoneme_to_index = dict()
		self.index_to_phoneme = list()
		index = 1
		#self.phoneme_to_index['<start>'] = 0
		#self.index_to_phoneme.append('<start>')
		self.phoneme_to_index['<end>'] = 0
		self.index_to_phoneme.append('<end>')
		for phoneme, _ in count_dict.items():
			self.phoneme_to_index[phoneme] = index
			self.index_to_phoneme.append(phoneme)
			index += 1

		#self.input_list = [[]] #[np.zeros((135000, self.max_len), dtype=np.int32), np.zeros((135000, self.decoder_len+1), dtype=np.int32)]
		alpha_list = [[] for _ in range(major_batch_count)]
		phone_list = [[] for _ in range(major_batch_count)]

		for word, pro in self.arpabet.items():
			#Word and P to index array

			c_word = re.sub('[^a-zA-Z]', '', word)
			c_word = c_word.lower()#[::-1]
			lw = len(c_word)
			lp = len(pro[0])
			batch_b = (lw, lp)
			if batch_len[str(batch_b)] < 10:
				continue
			s = [self.letter_to_index[w] for w in list(c_word)]
			s.append(1)
			l1 = len(s)

			p = pro[0]
			#for p in pro:
			s2 = []
			for phoneme in p:
				if phoneme[-1] == '0' or phoneme[-1] == '1' or phoneme[-1] == '2':
					phoneme = phoneme[:-1]
				s2 += [self.phoneme_to_index[phoneme]]
			s2.append(1)
			l2 = len(s2)

			alpha_list[batch_to_id[str(batch_b)]].append(s)
			phone_list[batch_to_id[str(batch_b)]].append(s2)

		batch_a_list = []
		batch_p_list = []
		for a_i in alpha_list:
			l_a_i = len(a_i)
			if l_a_i < 50:
				max_batch_count = int(50 / l_a_i)
				append_arr = np.empty([50, np.array(a_i).shape[1]], dtype=np.int32)
				for i in range(max_batch_count):
					append_arr[i*l_a_i:i*l_a_i + l_a_i, :] = a_i
				if 50 % l_a_i != 0:
					append_arr[max_batch_count*l_a_i:, :] = a_i[:50-max_batch_count*l_a_i]
				batch_a_list.append(append_arr)
			else:
				max_batch_count = int(l_a_i / 50)
				for i in range(max_batch_count):
					batch_a_list.append(np.array(a_i[50*i: 50*i+50]))
				if l_a_i % 50 != 0:
					batch_a_list.append(np.array(a_i[-50:]))

		for p_i in phone_list:
			l_a_i = len(p_i)
			if l_a_i < 50:
				max_batch_count = int(50 / l_a_i)
				append_arr = np.empty([50, np.array(p_i).shape[1]], dtype=np.int32)
				for i in range(max_batch_count):
					append_arr[i*l_a_i:i*l_a_i + l_a_i, :] = p_i
				if 50 % l_a_i != 0:
					append_arr[max_batch_count*l_a_i:, :] = p_i[:50-max_batch_count*l_a_i]
				batch_p_list.append(append_arr)
			else:
				max_batch_count = int(l_a_i / 50)
				for i in range(max_batch_count):
					batch_p_list.append(np.array(p_i[50*i: 50*i+50]))
				if l_a_i % 50 != 0:
					batch_p_list.append(np.array(p_i[-50:]))

		self.input_list = [batch_a_list, batch_p_list]
		for b in batch_a_list:
			print(np.array(b).shape)
		print(len(batch_a_list))
		print(len(batch_p_list))

if __name__ == '__main__':
	bg = BatchGen()
	diction = {"lti": bg.letter_to_index, "itl": bg.index_to_letter, "pti": bg.phoneme_to_index, "itp": bg.index_to_phoneme, "input": bg.input_list}

	pickle.dump(diction, open("words_phonemes.p", "wb"), protocol=2)