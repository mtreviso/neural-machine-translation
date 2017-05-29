# nmt
Simple seq2seq model for neural machine translation

## How to use?

sudo python3 seq2seq.py :params:


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

