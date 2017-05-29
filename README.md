# neural-machine-translation
Simple seq2seq model for neural machine translation

## How to use?
```sh
python3 seq2seq.py :params:
```


## List of available `params` and their default value
```
source-file			path to source file
target-file			path to target file
test-file			path to test file
gold-file			path to gold file; used for evaluating bleu score
mode				train/test/both; if both or test please inform :test-file: [train]
emb-type			embedding type: word2vec, fasttext, wang2vec, glove [wor2vec]
emb-file			path to trained :emb-type: model
load				path to weights.hdf5 file
save				path to weights.hdf5 file
gpu					run on GPU instead of on CPU
```

## Requirements
- `:emb-type:` model (see `embeddings` dir).
- [Keras](https://github.com/fchollet/keras/) in order to build the encoder-decoder architecture.
- [Theano](https://github.com/Theano/Theano) in order to run on GPU.
- [nltk](http://nltk.org/) in order to tokenize source and target texts.


## References

[1] Sutskever, Ilya, Oriol Vinyals, Quoc V. Le: [*Sequence to sequence learning with neural networks*](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

[2] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio: [*Neural machine translation by jointly learning to align and translate*](https://arxiv.org/pdf/1409.0473v6.pdf)

