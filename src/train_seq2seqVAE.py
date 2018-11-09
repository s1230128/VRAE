import numpy as np
import chainer
import chainer.functions   as F
import chainer.iterators   as I
import chainer.optimizers  as O
import chainer.serializers as S
import yaml, pickle
import os, shutil
import matplotlib.pyplot as plt
import seq2seq
import seq2seqVAE



''' ファイルの読み込み '''
# 設定ファイル
with open('./config.yml', 'r+') as f: config = yaml.load(f)
# 加工済みデータファイル
with open('../pickle/train.pickle', 'rb') as f:  train = pickle.load(f)
with open('../pickle/test.pickle' , 'rb') as f:  test  = pickle.load(f)
with open('../pickle/vocab.pickle', 'rb') as f:  word_id, id_word = pickle.load(f)


''' 出力の準備 '''
dir = config['out']['save_dir']
save_interval = config['out']['save_interval']

os.makedirs(dir+'/model', exist_ok=True)
shutil.copy('config.yml', dir+'/config.yml')


''' 学習の設定 '''
n_epoch     = config['param']['n_epoch']
size_embed  = config['param']['size_embed']
size_hidden = config['param']['size_hidden']
size_batch  = config['param']['size_batch']
n_vocab     = len(id_word)

model = seq2seqVAE.Seq2SeqVAE(n_vocab, size_embed, size_hidden)
opt = O.Adam()
opt.setup(model)
opt.add_hook(chainer.optimizer.GradientClipping(5))


''' 本丸 '''
for i in range(n_epoch):
    '''
    Train
     - バッチ毎に入力
     - パディングあり(トレーニングデータ全体の最大長に合わせて)
    '''
    epoch_loss = 0
    for b in I.SerialIterator(train, size_batch, repeat=False, shuffle=True):
        # データの整形
        enc, dec = zip(*b)
        enc = np.array(enc, dtype='int32').T
        dec = np.array(dec, dtype='int32').T
        # loss の計算
        loss = model.loss(enc, dec)
        epoch_loss += loss.data / len(train)
        # 更新
        model.cleargrads() # 学習前に内部の勾配をフラットに
        loss.backward()
        opt.update()

    '''
    Test
     - １データ毎に入力(バッチサイズ = １)
     - パディングなし
    '''
    epoch_accu = 0
    for b in I.SerialIterator(test, 1, repeat=False, shuffle=False):
        # データの整形
        enc, dec = zip(*b)
        enc = np.array(enc, dtype='int32').T
        dec = np.array(dec, dtype='int32').T
        # accuracy の計算
        accu = model.accuracy(enc, dec)
        epoch_accu += accu / len(test)
        ts = model(enc, dec[:-1])

    print(" ".join(id_word[F.argmax(t).data] for t in ts))


    ''' 出力 & モデルの保存 '''
    message = '{:>3} | {:>8.5f} | {:>6.1%}'.format(i+1, epoch_loss, epoch_accu)
    print(message)
    with open(dir + '/log.txt', 'a') as f:
        f.write(message + '\n')

    if (i+1) % save_interval == 0:
        path = dir + '/model/epoch_' + str(i+1) + '.npz'
        S.save_npz(path, model)
