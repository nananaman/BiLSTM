from pathlib import Path
import MeCab
import json
import numpy as np

import chainer
from chainer import Variable, optimizers, Chain, iterators, training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L


def make_minibatch(batch, mecabTagger, vocab):
    # abstとtitleを抜き出す
    abstracts = [b[0] for b in batch]
    titles = [b[1] for b in batch]
    # 日本語をIDに変更
    absts = []
    ttls = []
    for abstract, title in zip(abstracts, titles):
        node = mecabTagger.parseToNode(abstract)
        node = node.next

        abst = []
        while node:
            word = node.surface
            node = node.next
            wid = vocab[word]
            abst.append(wid)
        absts.append(abst)

        node = mecabTagger.parseToNode(title)
        node = node.next

        #ttl = [vocab["<EOS>"]]
        ttl = []
        while node:
            word = node.surface
            node = node.next
            wid = vocab[word]
            ttl.append(wid)
        ttls.append(ttl)

    enc_words = [row for row in absts]
    enc_max = np.max([len(row) for row in enc_words])
    enc_words = np.array([[-1]*(enc_max - len(row)) +
                          row for row in enc_words], dtype='int32')
    enc_words = enc_words.T

    dec_words = [row for row in ttls]
    dec_max = np.max([len(row) for row in dec_words])
    dec_words = np.array([[-1]*(dec_max - len(row)) +
                          row for row in dec_words], dtype='int32')
    dec_words = dec_words.T

    return enc_words, dec_words


class Attention(Chain):
    def __init__(self, hidden_size, flag_gpu=0):
        '''
        Attentionのインスタンス化
        :param hidden_size : 隠れ層のサイズ
        :param flag_gpu : GPUを使うか否か
        '''
        super().__init__(
            # Encoderの中間ベクトルを隠れ層サイズのベクトルに変換
            eh=L.Linear(hidden_size, hidden_size),
            # Decoderの中間ベクトルを隠れ層サイズのベクトルに変換
            dh=L.Linear(hidden_size, hidden_size),
            # スカラーに変換
            hw=L.Linear(hidden_size, 1),
        )
        self.hidden_size = hidden_size
        if flag_gpu:
            # self.ARR = cuda.cupy
            pass
        else:
            self.ARR = np

    def __call__(self, es, h):
        '''
        Attentionの計算
        :param es : Encoderの中間ベクトルが記録されたリスト
        :param h : Decoderの中間ベクトルが記録されたリスト
        :return : Encoderの中間ベクトルの加重平均
        '''
        # hの次元を落とす
        h = Variable(h.data[0])
        batch_size = h.data.shape[0]
        # 重みを記録するリスト
        ws = []
        # 重みの合計値
        sum_w = Variable(self.ARR.zeros((batch_size, 1), dtype='float32'))
        # EncoderとDecoderの中間ベクトルを使って重みの計算
        for e in es:
            # 重みの計算
            w = F.tanh(self.eh(e) + self.dh(h))
            # 正規化
            w = F.exp(self.hw(w))
            # 記録
            ws.append(w)
            sum_w += w
        # 加重平均ベクトルの初期化
        att = Variable(self.ARR.zeros(
            (1, batch_size, self.hidden_size), dtype='float32'))
        for e, w in zip(es, ws):
            # 重みの正規化
            w /= sum_w
            # 重み * Encoderの中間ベクトルを、出力するベクトルに加算
            att += F.reshape(F.batch_matmul(e, w),
                             (1, batch_size, self.hidden_size))
        return att


class Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        '''
        Encoderのインスタンス化
        :param vocab_size : 使われる単語の種類数（語彙数）
        :param embed_size : 単語をベクトル化した際のサイズ
        :param hidden_size : 中間層のサイズ
        '''
        dropout_rate = 0.25
        n_layers = 1
        super().__init__(
            # 単語をベクトル化
            xe=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # 双方向LSTM
            bilstm=L.NStepBiLSTM(n_layers, embed_size,
                                 int(hidden_size/2), dropout_rate),
        )

    def __call__(self, xs):
        '''
        Encoderの動作
        :param x : 入力(wid)
        :param c : 内部メモリ
        :param h : 隠れ層
        :return 次の隠れ層、次の内部メモリ
        '''
        xs_f = []
        for x in xs:
            x = self.xe(Variable(x))
            xs_f.append(x)
        
        hy, cy, ys = self.bilstm(hx=None, cx=None, xs=xs_f)
        '''
        print('hy : {}'.format(hy.shape))
        print('cy : {}'.format(cy.shape))
        print('ys : {}'.format(len(ys)))
        print('ys[0] : {}'.format(ys[0].shape))
        return hy, cy
        '''
        return ys, cy


class Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, flag_gpu=0):
        '''
        Decoderのインスタンス化
        :param vocab_size : 語彙数
        :param embed_size : 単語ベクトルのサイズ
        :param hidden_size : 隠れ層のサイズ
        '''
        dropout_rate = 0.25
        n_layers = 1
        super().__init__(
            # 単語のベクトル化
            ye=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # 入力と同じサイズを出力するLSTM
            lstm=L.NStepLSTM(n_layers, embed_size,
                             hidden_size, dropout_rate),
            # 隠れ層のベクトルを単語ベクトルのサイズに変換する層
            he=L.Linear(hidden_size, embed_size),
            # 単語ベクトルを語彙数サイズのベクトルに変換
            ey=L.Linear(embed_size, vocab_size),
        )
        if flag_gpu:
            pass
        else:
            self.ARR = np

    def __call__(self, y, h, c, att):
        '''
        Decoderの動作
        :param y : Decoderに入力する単語
        :param c : 内部メモリ
        :param h : Decoderの中間ベクトル
        :param att : Attentionで得たEncoderの加重平均
        :return : 語彙数サイズのベクトル、次の隠れ層、次の内部メモリ
        '''
        # 入力された単語をベクトル化してtanh
        e = []
        for w in y:
            e.append(F.tanh(self.ye(w)))
        # LSTM
        hy, cy, ys = self.lstm(h, c, e)
        # hyの次元を落としたhy_0を作る
        hy_0 = Variable(self.ARR.array([h for h in hy.data[0]]))
        # 出力された中間ベクトルを単語ベクトルに、単語ベクトルを語彙サイズのベクトルに変換
        t = self.ey(F.tanh(self.he(hy_0)))
        return t, hy, cy


class seq2seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, mecabTagger, vocab, id2wd, flag_gpu=0):
        '''
        BiLSTMを用いたSeq2Seqのインスタンス化
        :param vocab_size : 語彙数
        :param embed_size : 単語ベクトルのサイズ
        :param hidden_size : 隠れ層のサイズ
        :param batch_size : バッチサイズ
        :param flag_gpu : GPUを使うか否か
        '''
        super().__init__(
            encoder=Encoder(vocab_size, embed_size, hidden_size),
            attention=Attention(hidden_size, flag_gpu),
            decoder=Decoder(vocab_size, embed_size, hidden_size),
        )
        self.mecabTagger = mecabTagger
        self.vocab = vocab
        self.id2wd = id2wd
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        if flag_gpu:
            pass
        else:
            self.ARR = np

    def encode(self, words):
        '''
        Encoderの計算
        :param words : 入力で使用する単語(ID)のリスト
        :return : 
        '''
        # エンコード
        h, c = self.encoder(words)
        # Encoderの中間ベクトル
        self.es = h
        # 内部メモリと中間ベクトルの初期化
        self.c = Variable(self.ARR.zeros((1, self.batch_size, self.hidden_size), dtype='float32'))
        self.h = Variable(self.ARR.zeros((1, self.batch_size, self.hidden_size), dtype='float32'))
        #self.c = None
        #self.h = None

    def decode(self, w):
        '''
        Decoderの計算
        :param w : 入力する単語
        :return : 予測単語
        '''
        # AttentionでEncoderの加重平均を計算
        att = self.attention(self.es, self.h)
        # 予測
        t, self.h, self.c = self.decoder(w, self.h, self.c, att)
        return t

    def reset(self):
        '''
        インスタンス変数の初期化
        :return :
        '''
        self.c = Variable(self.ARR.zeros(
            (self.batch_size, self.hidden_size), dtype='float32'))
        self.h = Variable(self.ARR.zeros(
            (self.batch_size, self.hidden_size), dtype='float32'))
        self.zerograds

    def __call__(self, batch):
        '''
        Lossの計算
        :param : batch 入力ミニバッチ
        :return : loss
        '''
        # 勾配の初期化
        self.reset()

        abstracts, titles = make_minibatch(batch, self.mecabTagger, self.vocab)
        # np or cp に変更
        x_list = [self.ARR.array(x, dtype='int32') for x in abstracts]
        t_list = [self.ARR.array(x, dtype='int32') for x in titles]
        # encode
        self.encode(x_list)
        # lossの初期化
        loss = Variable(self.ARR.zeros((), dtype='float32'))
        # <EOS>をデコーダーに読み込ませる準備
        #t = Variable(self.ARR.array([self.vocab["<EOS>"] for _ in range(self.batch_size)], dtype='int32'))
        t = [Variable(self.ARR.array([self.vocab["<EOS>"]], dtype='int32')) for _ in range(self.batch_size)]
        # Decoderの計算
        for w in t_list:
            # decode
            y = self.decode(t)
            # 正解をVariableに変更
            t = Variable(w)
            # lossを計算
            loss += F.softmax_cross_entropy(y, t)
        return loss

    def predict(self, batch):
        '''
        Lossの計算
        :param : batch 入力ミニバッチ
        :return : loss
        '''
        # 勾配の初期化
        self.reset()
        # abstを抜き出す
        abstracts = [b[0] for b in batch]
        # 日本語をIDに変更
        absts = []
        for abstract in abstracts:
            node = self.mecabTagger.parseToNode(abstract)
            node = node.next

            abst = []
            while node:
                word = node.surface
                node = node.next
                wid = self.vocab[word]
                abst.append(wid)
            absts.append(abst)

            node = self.mecabTagger.parseToNode(title)
            node = node.next

        # np or cp に変更
        x_list = [self.AAA.array(x, dtype='int32') for x in absts]
        # encode
        self.encode(x_list)
        # <eos>をDecoderに入れる準備
        y = Variable(ARR.array([self.vocab["<EOS>"]
                                for _ in range(batch_size)], dtype='int32'))
        outs = []
        # ループ数のカウンタを初期化
        loop = 0
        # <EOS>が出るか30単語出力したら終了
        while(wid != self.vocab["<EOS>"]) and (loop <= 30):
            # 1語ずつdecode
            y = self.decode(y)

            wid = np.argmax(F.softmax(self.W(h)).data[0])
            out = [self.id2wd[wid]]
            loop = 0
            while(wid != self.vocab["<EOS>"]) and (loop <= 30):
                x_k = self.embedx(
                    Variable(np.array([wid], dtype=np.int32)))
                h = self.H(x_k)
                wid = np.argmax(F.softmax(self.W(h)).data[0])
                out.append(self.id2wd[wid])
                loop += 1
            outs.append(out)
        return outs
