import numpy as np
from chainer import Chain, Variable
from chainer import functions as F
from chainer import links     as L
import yaml
import nltk
import matplotlib.pyplot as plt



class Seq2Seq(Chain):

    def __init__(self, size_vocab, size_embed=300, size_hidden=300):
        '''
        : arg size_vocab  : 使われる単語の語彙数
        : arg size_embed  : 単語をベクトル表現した時のサイズ
        : arg size_hidden : 隠れ層のサイズ
        '''
        super(Seq2Seq, self).__init__()
        with self.init_scope():
            # encoder
            self.enc_embed = L.EmbedID(size_vocab, size_embed, ignore_label=-1)
            self.enc_lstm  = L.LSTM(size_embed, size_hidden)
            # decoder
            self.dec_embed = L.EmbedID(size_vocab, size_embed, ignore_label=-1)
            self.dec_lstm  = L.LSTM(size_embed, size_hidden)
            self.dec_out   = L.Linear(size_hidden, size_vocab)


    def encode(self, xs):
        for x in xs:
            e = F.tanh(self.enc_embed(x))
            h = self.enc_lstm(e)


    def decode(self, xs):
        ts = []
        for x in xs:
            e = F.tanh(self.dec_embed(x))
            h = self.dec_lstm(e)
            t = self.dec_out(h) #隠れ層を予測単語ベクトル(size_batch, size_vocab)に変換
            ts.append(t)
            if F.argmax(t).data == 1: break #出力が＜END＞なら終了

        return ts


    def __call__(self, enc, dec):
        '''
        : arg enc [(size_batch)]            : encode用の単語idの時系列データ
        : arg dec [(size_batch)]            : decode用の単語idの時系列データ
        : ret ts  [(size_batch, size_vocab] : 予測単語の時系列データ
        '''
        # model内で使用するLSTMの内部状態をリセット
        self.reset()

        self.encode(enc)
        self.dec_lstm.h = self.enc_lstm.h #encoderの隠れ層をdecoderに受け渡し
        ts = self.decode(dec)

        return ts


    def loss(self, enc, dec):
        ts = self(enc, dec[:-1])
        loss = sum(F.softmax_cross_entropy(t, w) for t, w in zip(ts, dec[1:]))

        return loss


    def accuracy(self, enc, dec):
        ts = self(enc, dec[:-1])

        accu = accu / len(ts)

        return accu


    def reset(self):
        """
        内部メモリの初期化
        """
        self.enc_lstm.reset_state()
        self.dec_lstm.reset_state()
