# extended-embedding-pub

Sentiment Analysis by radical(of Japanese kanji character)-level encoder.

This encoder refers to following paper.
- https://www.jstage.jst.go.jp/article/tjsai/33/4/33_D-I23/_pdf/-char/ja)

## Environment

- OS: Ubuntu 16.04.4 LTS
- GPU: GeForce GTX 1080
- Python: 3.6.5
- Chainer: 4.2.0
- MeCab: 0.996 （split sentence to words）

## Usage

Directory configuration as follows.
- create_vocab.py
- radical_classifier.py
- char2radical.py
- datasets
  - radical_dic.csv
  - set your train/test datasets


Make vocabulary-dictionary of your datasets with the ./datasets/radical_dic.csv
```:bash
python create_vocab.py
```

Edit Hyper-parameter, Run training script with the generated file.
```:bash
python radical_classifier.py
```

## Licence

[MIT](https://github.com/Nextremer/extended-embedding-pub/blob/master/LICENSE)

## Author

[syunna5](https://github.com/syunna5)
