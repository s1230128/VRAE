import numpy as np
from chainer import Chain, Variable
from chainer import functions as F
from chainer import links     as L
import yaml
import nltk
import matplotlib.pyplot as plt



class Seq2SeqVAE(Chain):

    def __init__(self, size_vocab, size_embed, size_hidden):
        '''
        : arg size_vocab  : 使われる単語の語彙数
        : arg size_embed  : 単語をベクトル表現した時のサイズ
        : arg size_hidden : 隠れ層のサイズ
        : arg size_batch  : バッチのサイズ
        '''
        super(Seq2SeqVAE, self).__init__()
        self.size_vocab = size_vocab
        with self.init_scope():
            # encoder
            self.enc_embed = L.EmbedID(size_vocab, size_embed, ignore_label=-1)
            self.enc_rnn   = L.GRU(size_embed, size_hidden)
            self.h_mu      = L.Linear(size_hidden, size_hidden)
            self.h_ln      = L.Linear(size_hidden, size_hidden)
            # decoder
            self.dec_embed = L.EmbedID(size_vocab, size_embed, ignore_label=-1)
            self.dec_rnn   = L.GRU(size_embed, size_hidden)
            self.dec_out   = L.Linear(size_hidden, size_vocab)


    def encode(self, xs):
        """
        エンコーダーの処理
        : arg xs [(size_batch)] : 単語IDの時系列データ
        """
        # 内部メモリの初期化
        self.enc_rnn.reset_state()

        for x in xs:
            e = F.tanh(self.enc_embed(x))
            h = self.enc_rnn(e)

        mu = self.h_mu(h)
        ln = self.h_ln(h)

        return mu, ln


    def decode(self, xs):
        """
        デコーダーの処理
        : arg xs [(size_batch)]             : 単語IDの時系列データ
        : ret ts [(size_batch, size_vocab)] : 予測単語ベクトルの時系列データ
        """
        # 内部メモリの初期化
        self.dec_rnn.reset_state()

        ts = []
        for x in xs:
            e = F.tanh(self.dec_embed(x))
            h = self.dec_rnn(e)
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
        # encode
        mu, ln = self.encode(enc)
        # encodeの結果をdecodeに受け渡し
        self.dec_rnn.h = mu
        # decode
        ts = self.decode(dec)

        return ts


    def loss(self, enc, dec, C=1.0, k=1):
        batchsize = len(enc[0])

        # 入力から、平均ベクトル、分散ベクトルの計算
        mu, ln = self.encode(enc)
        # 復元誤差の計算　Reconstraction Error
        reconst_loss = 0
        for _ in range(k):
            z = F.gaussian(mu, ln)
            self.dec_rnn.h = z
            ts = self.decode(dec)

            for d, t in zip(dec[1:], ts):
                reconst_loss += F.bernoulli_nll(np.eye(self.size_vocab)[d], t)

        reconst_loss = reconst_loss / (k * batchsize * len(ts))
        latent_loss  = C * F.loss.vae.gaussian_kl_divergence(mu, ln) / batchsize

        loss = reconst_loss + latent_loss

        return loss


    def accuracy(self, enc, dec):
        ts = self(enc, dec[:-1])
        accu = sum(F.accuracy(t, w).data for t, w in zip(ts, dec[1:]))
        accu = accu / len(ts)

        return accu
