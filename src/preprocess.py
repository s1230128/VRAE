import numpy as np
import yaml, pickle
import nltk
import chainer.datasets as D



# データの読み込み
with open('config.yml', 'r+') as f: config = yaml.load(f)

max_words = config['in']['max_words']
n_train = config['in']['n_train']
n_test  = config['in']['n_test']

with open(config['in']['train_enc']) as f: train_enc = [f.readline() for _ in range(n_train)]
with open(config['in']['train_dec']) as f: train_dec = [f.readline() for _ in range(n_train)]
with open(config['in']['test_enc' ]) as f: test_enc  = [f.readline() for _ in range(n_test)]
with open(config['in']['test_dec' ]) as f: test_dec  = [f.readline() for _ in range(n_test)]


# 文章を単語のリストに変換
train_enc = [nltk.word_tokenize(t) for t in train_enc]
train_dec = [nltk.word_tokenize(t) for t in train_dec]
test_enc  = [nltk.word_tokenize(t) for t in test_enc ]
test_dec  = [nltk.word_tokenize(t) for t in test_dec ]


#
train = [(e, d) for e, d in zip(train_enc, train_dec) if len(e) <= max_words and len(d) <= max_words]
test  = [(e, d) for e, d in zip(test_enc , test_dec ) if len(e) <= max_words and len(d) <= max_words]
train_enc, train_dec = zip(*train)
test_enc , test_dec  = zip(*test)


# word -> id の辞書の作成
word_id = {'<START>':0, '<END>':1}
for t in train_enc + train_dec + test_enc + test_dec:
    for w in t:
        if w not in word_id: word_id[w] = len(word_id)


# id -> word の辞書と見せかけてリスト
id_word = ['None'] * len(word_id)
for k, v in word_id.items(): id_word[v] = k


# パディング用の最大単語数
max_enc = max(len(d) for d in train_enc)
max_dec = max(len(d) for d in train_dec)


# 単語からIDに & 特殊単語を挿入(<START>:0  <END>:1  <PAD>:-1)
train_enc = [      [word_id[w] for w in t]       + [-1]*(max_enc-len(t))  for t in train_enc]
train_dec = [[0] + [word_id[w] for w in t] + [1] + [-1]*(max_dec-len(t))  for t in train_dec]
test_enc  = [      [word_id[w] for w in t]        for t in test_enc]
test_dec  = [[0] + [word_id[w] for w in t] + [1]  for t in test_dec]


train = D.TupleDataset(train_enc, train_dec)
test  = D.TupleDataset(test_enc , test_dec )


# 保存
with open('../pickle/train.pickle', 'wb') as f: pickle.dump(train, f)
with open('../pickle/test.pickle' , 'wb') as f: pickle.dump(test , f)
with open('../pickle/vocab.pickle', 'wb') as f: pickle.dump((word_id, id_word), f)
