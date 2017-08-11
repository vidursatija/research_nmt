import nltk
import re
import numpy as np
import pickle

class BatchGen():
	def __init__(self):
		self.arpabet = nltk.corpus.cmudict.dict()
		count_dict = dict()

		for key, value in self.arpabet.items():
			for p in value:

				for phoneme in p:
					if phoneme[-1] == '0' or phoneme[-1] == '1' or phoneme[-1] == '2':
						phoneme = phoneme[:-1]
					try:
						count_dict[str(phoneme)] += 1
					except:
						count_dict[str(phoneme)] = 1

		self.letter_to_index = dict()
		self.index_to_letter = list()
		self.letter_to_index['<start>'] = 0
		self.index_to_letter.append('<start>')
		self.letter_to_index['<end>'] = 1
		self.index_to_letter.append('<end>')
		for index in range(0, 26):
			self.letter_to_index[chr(index+97)] = index+2
			self.index_to_letter.append(chr(index+97))

		self.phoneme_to_index = dict()
		self.index_to_phoneme = list()
		index = 2
		self.phoneme_to_index['<go>'] = 0
		self.index_to_phoneme.append('<go>')
		self.phoneme_to_index['<end>'] = 1
		self.index_to_phoneme.append('<end>')
		for phoneme, _ in count_dict.items():
			self.phoneme_to_index[phoneme] = index
			self.index_to_phoneme.append(phoneme)
			index += 1

		self.input_list = [[], []] #[np.zeros((135000, self.max_len), dtype=np.int32), np.zeros((135000, self.decoder_len+1), dtype=np.int32)]

		for word, pro in self.arpabet.items():
			#Word and P to index array
			c_word = re.sub('[^a-zA-Z]', '', word)
			c_word = c_word.lower()#[::-1]
			s = [0] + [self.letter_to_index[w] for w in list(c_word)]
			s.append(1)
			l1 = len(s)

			p = pro[0]
			#for p in pro:
			s2 = [0]
			for phoneme in p:
				if phoneme[-1] == '0' or phoneme[-1] == '1' or phoneme[-1] == '2':
					phoneme = phoneme[:-1]
				s2 += [self.phoneme_to_index[phoneme]]
			s2.append(1)
			l2 = len(s2)

			self.input_list[0].append(s)
			self.input_list[1].append(s2)

		print(len(self.index_to_phoneme))

if __name__ == '__main__':
	bg = BatchGen()
	diction = {"lti": bg.letter_to_index, "itl": bg.index_to_letter, "pti": bg.phoneme_to_index, "itp": bg.index_to_phoneme, "input": bg.input_list}

	pickle.dump(diction, open("words_phonemes.p", "wb"), protocol=2)