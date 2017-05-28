import sys
import numpy as np
np.random.seed(42)

from keras.models import Sequential
from keras.layers import *
from utils import vectorize, column_matrix


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

	def train(self, X, Y, nb_epoch, step=1000):
		for e in range(nb_epoch):
			indices = np.arange(len(X))
			np.random.shuffle(indices)
			X, Y = X[indices], Y[indices]
			for i in range(0, len(X), step):
				sys.stdout.write('Training model: epoch {}th {}/{} samples \r'.format(e+1, i, len(X)))
				sys.stdout.flush()
				i_end = len(X) if i + step >= len(X) else i + step
				Y_vec = vectorize(Y[i:i_end], one_hot_dim=len(self.target_vocab))
				self.model.fit(X[i:i_end], Y_vec, batch_size=self.batch_size, epochs=1, verbose=0)
			print('')
	
	def test(self, X, step=1000):
		assert(self.model != None)
		source_idx_word = dict(zip(self.source_vocab.values(), self.source_vocab.keys()))
		target_idx_word = dict(zip(self.target_vocab.values(), self.target_vocab.keys()))
		sequences = []
		for i in range(0, len(X), step):
			sys.stdout.write('Prediction %d of %d \r' % (i+1, len(X)))
			sys.stdout.flush()
			i_end = len(X) if i + step >= len(X) else i + step
			sample = X[i:i_end]
			preds = self.model.predict_on_batch(np.matrix(sample))
			preds = np.argmax(preds, axis=-1)
			for prediction in preds:
				sequences.append([target_idx_word[index] for index in prediction if index > 0])
			# original = ' '.join([source_idx_word[index] for index in sample if index > 0])
			# translated = ' '.join(sequences[-1]) 
			# print('%d: %s, %s', (i+1, original, translated))
		print('')
		return sequences