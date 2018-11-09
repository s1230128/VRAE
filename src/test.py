import numpy as np
import chainer
import chainer.functions   as F
import chainer.iterators   as I
import chainer.optimizers  as O
import chainer.serializers as S
import yaml, pickle
import os, shutil
import nltk
import matplotlib.pyplot as plt
import seq2seq



''' ファイルの読み込み '''
# 設定ファイル
with open('config.yml', 'r+') as f: config = yaml.load(f)
with open('../pickle/vocab.pickle', 'rb') as f:  word_id, id_word = pickle.load(f)


''' 学習の設定 '''
n_epoch     = config['param']['n_epoch']
size_batch  = config['param']['size_batch']
size_embed  = config['param']['size_embed']
size_hidden = config['param']['size_hidden']
n_hidden    = config['param']['n_hidden']
n_vocab     = len(id_word)

model = seq2seq.Seq2Seq(n_vocab, size_embed, size_hidden)
S.load_npz('../result/lstm_shakespeare/model/epoch_50.npz', model)


while True:
    print('in  : ', end='')

    # データの整形
    text = input()
    text = [word_id[w] for w in nltk.word_tokenize(text)]

    enc = np.array([text], dtype='int32').T
    dec = np.array([[word_id['<START>']] + text + [word_id['<END>']]], dtype='int32').T


    # forward 処理
    ts = model(enc, dec[:-1])
    print('out :', ' '.join(id_word[F.argmax(t).data] for t in ts))
