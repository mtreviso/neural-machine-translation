import sys
import argparse
import numpy as np
np.random.seed(42)

from dataset import DataSet
from embeddings import AvailableEmbeddings
from model import EncoderDecoder


# FNAME_SOURCE = 'data/pt-en/europarl-v7.pt-en.pt'
# FNAME_TARGET = 'data/pt-en/europarl-v7.pt-en.en'

MAX_LEN = 50 # 300
MAX_VOCAB_SIZE = 1000 # 20000
BATCH_SIZE = 128
NB_EPOCH = 1
FNAME_WEIGHTS = 'data/encoder_decoder.hdf5'
EMB_TYPE = 'word2vec'
EMB_FILE = '/media/treviso/SAMSUNG/Embeddings/ptbr/word2vec/pt_word2vec_sg_300.emb'


def load_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--source-file', type=str, help='path to source file', required=True)
	parser.add_argument('-t', '--target-file', type=str, help='path to target file', required=True)
	parser.add_argument('--test-file', type=str, help='path to test file')
	parser.add_argument('--mode', type=str, help='train or test; if test please inform :test-file:', default='train')
	parser.add_argument('--emb-type', type=str, help='emb type', default=EMB_TYPE)
	parser.add_argument('--emb-file', type=str, help='emb file', default=EMB_FILE)
	parser.add_argument('--load', action='store_true', help='load weights')
	parser.add_argument('--save', action='store_true', help='save weights')
	parser.add_argument('--gpu', action='store_true', help='run on GPU instead of on CPU')
	return parser.parse_args()


def load_emb_model(etype, fname):
	emb_model = AvailableEmbeddings.get(etype)()
	emb_model.load(fname)
	return emb_model

def run(options):

	print('Loading dataset...')
	dataset = DataSet(source=options.source_file, target=options.target_file, max_len=MAX_LEN)
	dataset.info()

	print('Building vocab...')
	dataset.build_vocab(max_vocab_size=MAX_VOCAB_SIZE)

	print('Transforming data...')
	X, Y = dataset.to_matrix(pad=True)

	print('Loading embeddings...')
	emb_model = load_emb_model(options.emb_type, options.emb_file)


	ed = EncoderDecoder(dataset.source_word_idx, dataset.target_word_idx, emb_model=emb_model,
						input_length=len(X[0]), inner_length=len(Y[0]), batch_size=BATCH_SIZE)

	print('Building model...')
	ed.build()

	if options.load:
		print('Loading weights...')
		ed.load_weights(FNAME_WEIGHTS)

	if options.mode == 'train':
		print('Training model...')
		ed.train(X, Y, nb_epoch=NB_EPOCH)
	else:
		print('Testing model...')
		test_dataset = TestDataSet(options.test_file, dataset.source_word_idx, max_len=MAX_LEN)
		X_test = test_dataset.to_matrix(pad=True)
		ed.load_weights(FNAME_WEIGHTS) # force load weights
		ed.test(X_test)

	if options.save:
		print('Saving weights...')
		ed.save_weights(FNAME_WEIGHTS)






if __name__ == '__main__':

	options = load_options()

	# use GPU?
	if options.gpu == '':
		import theano.sandbox.cuda
		theano.sandbox.cuda.use('gpu')

	run(options)