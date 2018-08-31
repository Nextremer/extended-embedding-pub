# coding:utf-8

"""
Written_by: Sugimoto Shun
Created_at: 2018-08-31
Abstract: 部首レベルのエンコーディングを行い，極性判定を行うプログラム．
          漢字に対応する部首の辞書（radical_dic.csv）およびデータセットの
          ボキャブラリ辞書（create_vocab.pyを用いて作成）が必要となる．
"""

import os
import MeCab
import numpy as np
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, serializers, cuda
import matplotlib
matplotlib.use('Agg')  # 外部GPU環境を利用するときのmatplot設定(この位置に書く)
import matplotlib.pyplot as plt


xp = cuda.cupy
current_dir = os.path.dirname(os.path.abspath(__file__))


class ParameterClass(object):
    '''学習時のパラメータや変数'''
    def __init__(self):
        self.use_gpu = True
        self.class_num  = 2
        self.epochs     = 50
        self.batch_size = 50

        self.optimizer_type = 'RMSprop'
        self.leaning_rate   = 0.001

        self.WD_flag = False  # Weight Decay
        self.rate    = 10e-6
        self.GC_flag   = False  # Gradient Clipping
        self.threshold = 1.0

        self.bnorm_flag = False

        self.embed_dim = 45
        self.word_len  = 40 # 1文に含まれる単語数

        # この二つは変更しない方が良い
        self.character_len = 3  # 1単語に含まれる文字数
        self.radical_len   = 3  # 1文字に含まれる部首数

        self.document_len = (self.radical_len *
                             self.character_len *
                             self.word_len)


class Model(chainer.Chain):
    '''ネットワークモデル'''
    def __init__(self, vocab_size, ParameterClass):
        self.a = ParameterClass
        self.vocab_size = vocab_size
        # 特徴を獲得するため複数フィルター(部首単位，文字単位)を適用
        self.cnn_window_sizes = [1, 2, 3, 3, 6, 9]
        self.cnn_stride_sizes = [1, 1, 1, 3, 3, 3]
        # ウインドウとストライドに応じたフィルター数を適用
        self.cnn_filter_nums = [int(50*(w/r)) for w, r in
                                zip(self.cnn_window_sizes,
                                    self.cnn_stride_sizes)]
        # poolingウインドウは畳み込み時のウインドウとストライドに依存(1単語ごと)
        self.pooling_window_sizes = [
            int((self.a.radical_len * self.a.character_len - w) / r + 1)
            for w, r in zip(self.cnn_window_sizes, self.cnn_stride_sizes)]

        initializer = chainer.initializers.HeNormal()

        super(Model, self).__init__()
        with self.init_scope():
            # Embedding
            self.embed = L.EmbedID(
                            self.vocab_size,
                            self.a.embed_dim,
                            initialW=initializer)

            # Convolution (6種類)
            self.conv0 = L.Convolution2D(
                            1,
                            self.cnn_filter_nums[0],
                            ksize=(self.cnn_window_sizes[0], self.a.embed_dim),
                            stride=(self.cnn_stride_sizes[0], self.a.embed_dim))
            self.conv1 = L.Convolution2D(
                            1,
                            self.cnn_filter_nums[1],
                            ksize=(self.cnn_window_sizes[1], self.a.embed_dim),
                            stride=(self.cnn_stride_sizes[1], self.a.embed_dim))
            self.conv2 = L.Convolution2D(
                            1,
                            self.cnn_filter_nums[2],
                            ksize=(self.cnn_window_sizes[2], self.a.embed_dim),
                            stride=(self.cnn_stride_sizes[2], self.a.embed_dim))
            self.conv3 = L.Convolution2D(
                            1,
                            self.cnn_filter_nums[3],
                            ksize=(self.cnn_window_sizes[3], self.a.embed_dim),
                            stride=(self.cnn_stride_sizes[3], self.a.embed_dim))
            self.conv4 = L.Convolution2D(
                            1,
                            self.cnn_filter_nums[4],
                            ksize=(self.cnn_window_sizes[4], self.a.embed_dim),
                            stride=(self.cnn_stride_sizes[4], self.a.embed_dim))
            self.conv5 = L.Convolution2D(
                            1,
                            self.cnn_filter_nums[5],
                            ksize=(self.cnn_window_sizes[5], self.a.embed_dim),
                            stride=(self.cnn_stride_sizes[5], self.a.embed_dim))
            self.cnn_output_dim = sum(self.cnn_filter_nums)

            # pooling前のBatchNormalization
            self.bnorm0 = L.BatchNormalization(self.cnn_filter_nums[0])
            self.bnorm1 = L.BatchNormalization(self.cnn_filter_nums[1])
            self.bnorm2 = L.BatchNormalization(self.cnn_filter_nums[2])
            self.bnorm3 = L.BatchNormalization(self.cnn_filter_nums[3])
            self.bnorm4 = L.BatchNormalization(self.cnn_filter_nums[4])
            self.bnorm5 = L.BatchNormalization(self.cnn_filter_nums[5])

            # Highway1
            self.hw1 = L.Highway(
                        self.cnn_output_dim,
                        activate=F.tanh,
                        init_Wh=initializer,
                        init_Wt=initializer)

            # BiLSTM
            self.bi_lstm_dim = self.cnn_output_dim * 2
            self.bi_lstm = L.NStepBiLSTM(
                            n_layers=1,
                            in_size=self.cnn_output_dim,
                            out_size=self.cnn_output_dim,
                            dropout=0.0)

            # Higiway2 + Soft Attention
            self.hw2 = L.Highway(
                        self.bi_lstm_dim,
                        activate=F.tanh,
                        init_Wh=initializer,
                        init_Wt=initializer)
            self.u_a = chainer.Parameter(initializer,
                                         (1, self.bi_lstm_dim))
            # output (+ BatchNormalization)
            self.fc = L.Linear(self.bi_lstm_dim, 2, initialW=initializer)
            self.bnorm_last = L.BatchNormalization(2)

    def __call__(self, mb, t):
        '''Lossの計算'''
        loss = F.softmax_cross_entropy(self.fwd(mb), t)
        return loss

    def fwd(self, mb):
        '''ネットワークの出力を計算'''
        # mb (mb_size, channel_size(1), radical_sequence)

        # radical_emb (mb_size, channel_size, radical_sequence, embed_dim)
        radical_emb = self.embed(mb)

        # cnn_output (mb_size, vecter_dim(output_channel), word_step, 1)
        cnn_output = self.concat_cnn(radical_emb)[:, :, :, 0]
        mb_size = cnn_output.shape[0]
        vector_dim = cnn_output.shape[1]
        word_step_len = cnn_output.shape[2]

        # -> cnn_output (mb_size * word_step, vecter_dim) -> Highway1
        cnn_output = F.swapaxes(cnn_output, 1, 2)
        cnn_output = F.reshape(cnn_output, (mb_size*word_step_len, vector_dim))
        hw_out = self.hw1(cnn_output)
        hw_out = F.reshape(hw_out, (mb_size, word_step_len, vector_dim))

        # ndarray -> list (LSTM_layer needs list-type for each data)
        hw_out = [hw_out[i, :, :] for i in range(len(mb))]

        # BiLSTM
        hy, cy, ys = self.bi_lstm(hx=None, cx=None, xs=hw_out)

        # list -> ndarray (axis=0)
        h_i = F.concat([i[None, :, :] for i in ys], axis=0)

        # Highway2
        vector_dim = vector_dim * 2
        ys = F.reshape(h_i, (mb_size*word_step_len, vector_dim))
        u_i = self.hw2(ys)
        u_i = F.reshape(u_i, (mb_size, word_step_len, vector_dim))

        # Soft Attention
        u_a = F.broadcast_to(self.u_a, (u_i.shape[0], vector_dim))
        a_i = F.batch_matmul(u_i, u_a)[:, :, 0]
        a_i = F.softmax(a_i)[0, :]
        # h_i -> (mb_size, vector, word_step), a_i -> (mb_size, word_step)
        h_i = F.swapaxes(h_i, 1, 2)
        a_i = F.broadcast_to(a_i, (h_i.shape[0], a_i.shape[0]))
        # バッチごとにh_iとa_iの行列積
        z = F.batch_matmul(h_i, a_i)[:, :, 0]

        # output
        y = self.fc(z)
        if self.a.bnorm_flag is True:
            y = self.bnorm_last(y)

        return y

    def concat_cnn(self, emb):
        '''CNN層の出力を返す'''
        output = []
        for i in range(self.a.word_len):
            # 単語単位(radical_len*character_len, embed_dim)でCNN
            input = emb[:,
                        :,
                        i:i+self.a.radical_len*self.a.character_len,
                        :]
            # Batch_Normalization
            if self.a.bnorm_flag is True:
                output.append(F.concat([
                    F.max_pooling_2d(F.relu(self.bnorm0(self.conv0(input))),
                                     (self.pooling_window_sizes[0], 1)),
                    F.max_pooling_2d(F.relu(self.bnorm1(self.conv1(input))),
                                     (self.pooling_window_sizes[1], 1)),
                    F.max_pooling_2d(F.relu(self.bnorm2(self.conv2(input))),
                                     (self.pooling_window_sizes[2], 1)),
                    F.max_pooling_2d(F.relu(self.bnorm3(self.conv3(input))),
                                     (self.pooling_window_sizes[3], 1)),
                    F.max_pooling_2d(F.relu(self.bnorm4(self.conv4(input))),
                                     (self.pooling_window_sizes[4], 1)),
                    F.max_pooling_2d(F.relu(self.bnorm5(self.conv5(input))),
                                     (self.pooling_window_sizes[5], 1))
                ], axis=1))
            else:
                output.append(F.concat([
                    F.max_pooling_2d(F.relu(self.conv0(input)),
                                     (self.pooling_window_sizes[0], 1)),
                    F.max_pooling_2d(F.relu(self.conv1(input)),
                                     (self.pooling_window_sizes[1], 1)),
                    F.max_pooling_2d(F.relu(self.conv2(input)),
                                     (self.pooling_window_sizes[2], 1)),
                    F.max_pooling_2d(F.relu(self.conv3(input)),
                                     (self.pooling_window_sizes[3], 1)),
                    F.max_pooling_2d(F.relu(self.conv4(input)),
                                     (self.pooling_window_sizes[4], 1)),
                    F.max_pooling_2d(F.relu(self.conv5(input)),
                                     (self.pooling_window_sizes[5], 1))
                ], axis=1))

        output = F.concat(output, axis=2)
        return output


def train(a, x_train, y_train, x_test, y_test, batch_size, epochs, vocab_size):
    '''トレーニング・評価'''
    model = Model(vocab_size, a)

    # select optimizer
    if a.optimizer_type == 'SGD':
        optimizer = optimizers.SGD(lr=a.leaning_rate)
    elif a.optimizer_type == 'RMSprop':
        optimizer = optimizers.RMSprop(lr=a.leaning_rate, alpha=0.9)
    elif a.optimizer_type == 'Adam':
        optimizer = optimizers.Adam()
    optimizer.setup(model)

    # WeightDecay + GradientClipping
    if a.WD_flag:
        optimizer.add_hook(chainer.optimizer.WeightDecay(a.rate))
    if a.GC_flag:
        optimizer.add_hook(chainer.optimizer.GradientClipping(a.threshold))

    if a.use_gpu:
        gpu_device = 0
        cuda.get_device(gpu_device).use()
        model.to_gpu(gpu_device)

    losses, train_accuracies, test_accuracies = [], [], []

    for epoch in range(epochs):
        perm = np.random.permutation(len(x_train))
        for i in range(0, len(x_train), batch_size):
            x = Variable(x_train[perm[i:i + batch_size]])
            t = Variable(y_train[perm[i:i + batch_size]])
            model.cleargrads()
            loss = model(x, t)
            loss.backward()
            optimizer.update()
        losses.append(loss.data)

        # training-accuracyの計算
        predict = F.softmax(model.fwd(x_train)).data
        predict = xp.array([np.argmax(i) for i in predict], dtype=xp.int32)
        train_accuracies.append(sum([(1 if y == t else 0) for y, t in
                                zip(predict, y_train)]) / float((len(y_train))))
        # test-accuracyの計算
        predict = F.softmax(model.fwd(x_test)).data
        predict = xp.array([np.argmax(i) for i in predict], dtype=xp.int32)
        test_accuracies.append(sum([(1 if y == t else 0) for y, t in
                               zip(predict, y_test)]) / float((len(y_test))))
    ''' plot loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(current_dir + '/loss.png')
    plt.clf()
    '''

    # plot Accuracy
    plt.plot(train_accuracies, label="train")
    plt.plot(test_accuracies, label="test")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.ylim(0, 1.0)
    plt.savefig(current_dir + '/accuracy.png')
    plt.clf()
    print('test accuracy:', test_accuracies[-1])

    return model


def sentence2radical(a, sentences):
    '''
    入力された文を部首(radical)のIDシーケンスに変換して返す関数
    '''
    w_len = a.word_len
    c_len = a.character_len
    r_len = a.radical_len
    vocab = pd.read_csv(current_dir+'/datasets/vocab_dictionary.csv',
                        index_col=0)
    df = pd.read_csv(current_dir+'/datasets/radical_dic.csv', index_col=0)
    radical_id_sequence = []

    for sentence in sentences:
        sentence_id = np.zeros(a.document_len, dtype=np.int32)
        m = MeCab.Tagger('-Owakati')
        words = m.parse(sentence).split(' ')
        for i, word in enumerate(words[:w_len]):
            for j, character in enumerate(word[:2]):  # 各単語は3文字まで見る
                radicals = df[df['CHARACTER'] == character]['RADICAL'].values
                if radicals:
                    radicals = radicals[0].split(' ')[:2]  # 部首も3文字まで
                    for k, radical in enumerate(radicals):
                        id = vocab[vocab['RADICAL'] == radical].index[0]
                        sentence_id[i*r_len*c_len + j*c_len + k] = id+1
                        # 1を足す（0は何もない場合）
                else:
                    id = vocab[vocab['RADICAL'] == character].index[0]
                    sentence_id[i*r_len*c_len + j*c_len] = id+1
        radical_id_sequence.append(sentence_id)
    return xp.array(radical_id_sequence), len(vocab)+1


def make_dataset(a, file_path):
    '''データを学習に利用できる形式に変換する関数'''
    df = pd.read_excel(file_path, skiprows=0, header=0)
    sentences = np.array(df[u'表層'])
    x, vocab_size = sentence2radical(a, sentences)
    y = xp.array([1 if i >= 0.5 else 0 for i in df[u'正規化スコア']])
    x = x[:, None, :]  # channel軸の追加
    return x, y, vocab_size


def main(a):
    traing_data_path = current_dir + '/datasets/training_data.xlsx'
    test_data_path = current_dir + '/datasets/test_data.xlsx'

    # データ読み込み
    x_train, y_train, vocab_size = make_dataset(a, traing_data_path)
    x_test, y_test, _ = make_dataset(a, test_data_path)

    # 学習，評価
    model = train(a, x_train, y_train, x_test, y_test,
                  batch_size=a.batch_size, epochs=a.epochs,
                  vocab_size=vocab_size)

    # モデルの保存
    model.to_cpu()
    serializers.save_npz("model.npz", model)


if __name__ == '__main__':
    a = ParameterClass()
    main(a)
