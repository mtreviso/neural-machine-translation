import sys
import numpy as np
np.random.seed(42)

from keras.models import Sequential
from keras.layers import *
from utils import vectorize


class EncoderDecoder:

	def __init__(self, source_vocab, target_vocab, emb_model=None, input_length=None, inner_length=None, batch_size=64):
		self.source_vocab = source_vocab
		self.target_vocab = target_vocab
		self.emb_model = emb_model
		self.input_length = input_length
		self.inner_length = inner_length
		self.output_size = len(target_vocab)
		self.batch_size = batch_size
		self.model = None
		self._prepare_params()


	def _prepare_params(self):
		vocab_size = max(self.source_vocab.values()) + 1
		emb_dim = self.emb_model.dimensions
		limit = np.sqrt(6 / (vocab_size + emb_dim)) # glorot_uniform
		self.emb_weights = np.random.uniform(-limit, limit, size=(vocab_size, emb_dim))
		for word, index in self.source_vocab.items():
			self.emb_weights[index] = self.emb_model.get_vector(word)
		self.emb_vocab_size = self.emb_weights.shape[0]
		self.emb_size = self.emb_weights.shape[1]

	
	def build(self, num_layers=1, hidden_size=100, rnn='GRU'):
		RNN = LSTM if rnn == 'LSTM' else GRU
		self.model = Sequential()

		# Encoder
		self.model.add(Embedding(self.emb_vocab_size, self.emb_size, input_length=self.input_length, mask_zero=True, weights=[self.emb_weights]))
		self.model.add(RNN(hidden_size))
		self.model.add(RepeatVector(self.inner_length))

		# Decoder
		for _ in range(num_layers):
			self.model.add(RNN(hidden_size, return_sequences=True))
			self.model.add(TimeDistributed(Dense(len(self.target_vocab))))
			self.model.add(Activation('softmax'))

		self._compile()
		self._summary()
	
	def _compile(self):
		print('Compiling...')
		self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

	def _summary(self):
		self.model.summary()

	def save_weights(self, filename):
		self.model.save_weights(filename)

	def load_weights(self, filename):
		self.model.load_weights(filename)

	def train(self, X, Y, nb_epoch):
		i_end = 0
		for k in range(1, nb_epoch+1):
			# Shuffling the training data every epoch to avoid local minima
			indices = np.arange(len(X))
			np.random.shuffle(indices)
			X = X[indices]
			Y = Y[indices]

			# Training 1000 sequences at a time
			for i in range(0, len(X), 1000):
				sys.stdout.write('Training model: epoch {}th {}/{} samples \r'.format(k, i, len(X)))
				sys.stdout.flush()
				i_end = len(X) if i + 1000 >= len(X) else i + 1000
				Y_vec = vectorize(Y[i:i_end], one_hot_dim=len(self.target_vocab))
				self.model.fit(X[i:i_end], Y_vec, batch_size=self.batch_size, epochs=1, verbose=0)
	
	def test(X_test):
		assert(self.model != None)
		predictions = np.argmax(self.model.predict(X_test), axis=2)
		target_idx_word = dict(zip(self.target_vocab.values(), self.target_vocab.keys()))
		sequences = []
		for prediction in predictions:
			sequence = ' '.join([target_idx_word[index] for index in prediction if index > 0])
			print(sequence)
			sequences.append(sequence)
		np.savetxt('data/test_result', sequences, fmt='%s')