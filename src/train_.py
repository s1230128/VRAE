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



''' ファイルの読み込み '''
# 設定ファイル
with open('config.yml', 'r+') as f: config = yaml.load(f)
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
size_batch  = config['param']['size_batch']
size_embed  = config['param']['size_embed']
size_hidden = config['param']['size_hidden']
n_hidden    = config['param']['n_hidden']
n_vocab     = len(id_word)

model = seq2seq.Seq2Seq(n_vocab, size_embed, size_hidden)
opt = O.Adam()
opt.setup(model)
#opt.add_hook(chainer.optimizer.GradientClipping(5))


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
        print(enc)
        print(dec)
        # forward 処理
        ts = model(enc, dec[:-1])
        # loss の計算
        loss = sum(F.softmax_cross_entropy(t, w) for t, w in zip(ts, dec[1:]))
        epoch_loss += loss.data / (len(train) * len(ts))
        # 更新
        model.cleargrads() # 学習前に内部の勾配をフラットに
        loss.backward()
        opt.update()

    # テストをスキップ
    #if (i + 1) % test_interval != 0: continue

    '''
    Test
     - １データ毎に入力(バッチサイズ = １)
     - パディングなし
    '''
    epoch_accu = 0
    for b in I.SerialIterator(test, 1, repeat=False, shuffle=False):
        # データの整形
        enc, dec = zip(*b)
        # forward 処理
        ts = model(enc, dec[:-1])
        # accuracy の計算
        accu = sum(F.accuracy(t, w) for t, w in zip(ts, dec[1:]))
        epoch_accu += accu.data / (len(test) * len(ts))
        #print(" ".join(id_word[F.argmax(t).data] for t in ts))


    ''' 出力'''
    message = '{:>3} | {:>8.5f} | {:>6.1%}'.format(i+1, epoch_loss, epoch_accu)
    print(message)
    with open(dir + '/log.txt', 'a') as f:
        f.write(message + '\n')


# モデルの保存
S.save_npz(dir + '/model/epoch_' + str(i+1) + '.npz', model)
