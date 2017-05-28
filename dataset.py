import numpy as np
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from utils import unroll, pad_sequences

np.random.seed(42)


class DataSet():

	def __init__(self, source, target, max_len, normalize=False):
		self.data = [[], []]
		self.dist = [None, None]
		self.vocab = [None, None]
		self.normalize = normalize
		self.max_len = max_len
		self._load_data(source, target)



	def _load_data(self, source, target):
		with open(source, encoding='utf8') as f1, open(target, encoding='utf8') as f2:
			for s, t in zip(f1, f2):
				s, t = s.strip(), t.strip()
				# Further information about reversing text order:
				# Sequence to sequence learning with neural networks. 2014.
				# from: Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. 
				s = self._tokenize_source(s)[::-1]
				t = self._tokenize_target(t)[::-1]
				if len(s) <= self.max_len and len(t) <= self.max_len and len(s) > 0 and len(t) > 0:
					self.data[0].append(s)
					self.data[1].append(t)


	def _tokenize_source(self, text):
		return text.split()
		# create your own tokenizer
		# tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
		# if self.normalize:
		# 	return tokenizer.tokenize(self._normalize(text))
		# return tokenizer.tokenize(text)


	def _tokenize_target(self, text):
		return text.split()
		# create your own tokenizer
		# tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
		# if self.normalize:
		# 	return tokenizer.tokenize(self._normalize(text))
		# return tokenizer.tokenize(text)


	def _normalize(self, text):
		# add your own normalization rules
		return text

	def build_vocab(self, max_vocab_size=20000):
		self.dist[0] = FreqDist(unroll(self.data[0]))
		self.vocab[0] = list(map(lambda x: x[0], self.dist[0].most_common(max_vocab_size - 1)))
		self.dist[1] = FreqDist(unroll(self.data[1]))
		self.vocab[1] = list(map(lambda x: x[0], self.dist[1].most_common(max_vocab_size - 1)))

	def to_matrix(self, pad=False):
		self.source_word_idx = dict(zip(['PAD', 'UNK'] + self.vocab[0], range(len(self.vocab[0])+2)))
		self.target_word_idx = dict(zip(['PAD', 'UNK'] + self.vocab[1], range(len(self.vocab[1])+2)))
		X, Y = [], []
		for i in range(len(self.data[0])):
			sent_source, sent_target = self.data[0][i], self.data[1][i]
			f = lambda t: self.source_word_idx[t] if t in self.source_word_idx else self.source_word_idx['UNK']
			X.append(list(map(f, sent_source)))
			g = lambda t: self.target_word_idx[t] if t in self.target_word_idx else self.target_word_idx['UNK']
			Y.append(list(map(g, sent_target)))
		if pad:
			X = pad_sequences(X, mask_value=0)
			Y = pad_sequences(Y, mask_value=0)
		return X, Y

	def info(self):
		print('Source Nb sentences: {}'.format(len(self.data[0])))
		print('Source Nb words: {}'.format(sum(map(len, self.data[0]))))
		print('Target Nb sentences: {}'.format(len(self.data[1])))
		print('Target Nb words: {}'.format(sum(map(len, self.data[1]))))
		

