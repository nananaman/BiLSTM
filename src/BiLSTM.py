from pathlib import Path
import MeCab
import json
import numpy as np

import chainer
from chainer import Variable, optimizers, Chain, iterators, training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import cupy as cp


def make_minibatch(batch, mecabTagger, vocab, ARR):
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

        ttl = []
        while node:
            word = node.surface
            node = node.next
            wid = vocab[word]
            ttl.append(wid)
        ttl.append(vocab["<EOS>"])
        ttls.append(ttl)

    enc_words = absts
    dec_words = ttls
    '''
    enc_max = np.max([len(row) for row in enc_words])
    enc_words = np.array([[-1]*(enc_max - len(row)) +
                          row for row in enc_words], dtype='int32')
    enc_words = enc_words
    '''
    dec_max = np.max([len(row) for row in dec_words])
    dec_words = np.array([row + [-1]*(dec_max - len(row))
                          for row in dec_words], dtype='int32')
    dec_words = dec_words.T
    #'''

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
            eh=L.Linear(hidden_size, 1),
        )
        self.hidden_size = hidden_size
        if flag_gpu:
            self.ARR = cp
        else:
            self.ARR = np

    def __call__(self, hs, ht, x_list):
        '''
        Attentionの計算
        :param hs : Encoderの中間ベクトルが記録されたリスト
        :param ht : Decoderの中間ベクトルが記録されたリスト
        :return : 中間ベクトルht
        '''
        batch_size = len(x_list)
        ht = F.reshape(ht, (batch_size, 1, self.hidden_size))
        h = []
        for i in range(batch_size):
            h.append(Variable((hs[i].data * ht[i].data)))
        concat_h = F.concat(h, axis=0)
        attn = self.eh(concat_h)
        sections = np.cumsum([len(x) for x in x_list])
        split_attention = F.split_axis(attn, sections[:-1], axis=0)
        split_attention_pad = F.pad_sequence(split_attention, padding=-1024.)
        attn_softmax = F.softmax(split_attention_pad, axis=1)
        hs_pad = F.pad_sequence(hs)
        hs_pad_reshape = F.reshape(hs_pad, (-1, hs_pad.shape[-1]))

        r = F.reshape(attn_softmax, (-1, attn_softmax.shape[-1]))
        attn_softmax_reshape = F.broadcast_to(
            F.reshape(attn_softmax, (-1, attn_softmax.shape[-1])), hs_pad_reshape.shape)
        attention_hidden = hs_pad_reshape * attn_softmax_reshape

        attention_hidden_reshape = F.reshape(
            attention_hidden, (batch_size, -1, attention_hidden.shape[-1]))
        result = F.sum(attention_hidden_reshape, axis=1)

        '''
        # hの次元を落とす
        h = Variable(h.data[0])
        batch_size = h.data.shape[0]
        # 重みを記録するリスト
        ws = []
        # 重みの合計値を初期化
        sum_w = Variable(self.ARR.zeros((batch_size, 1), dtype='float32'))
        # EncoderとDecoderの中間ベクトルを使って重みの計算
        for e in es:
            print(e.shape)
            print(h.shape)
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
                             '''
        return F.reshape(result, (1, result.shape[0], result.shape[1]))

def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0, force_tuple=True)
    return exs

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
        '''
        xs_f = []
        for x in xs:
            x = self.xe(Variable(x))
            xs_f.append(x)
        '''
        xs_f = sequence_embed(self.xe, xs)

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
            self.ARR = cp
        else:
            self.ARR = np

    def __call__(self, y, ht, ct):
        '''
        Decoderの動作
        :param y : Decoderに入力する単語
        :param ht : Decoderの中間ベクトル
        :param ct : 内部メモリ
        :return : 語彙数サイズのベクトル、次の隠れ層、次の内部メモリ
        '''
        # 入力された単語をベクトル化してtanh
        #e = [F.tanh(self.ye(w)) for w in y]
        e = sequence_embed(self.ye, y)
        # LSTM
        ht, ct, _ = self.lstm(ht, ct, e)
        # hyの次元を落としたhy_0を作る
        # hy_0 = Variable(self.ARR.array(hy.data[0]))
        # hy_0 = Variable(self.ARR.array([h for h in hy.data[0]]))
        # hy_0 = Variable(np.array([h for h in hy.data[0]]))
        # 出力された中間ベクトルを単語ベクトルに、単語ベクトルを語彙サイズのベクトルに変換
        # t = self.ey(F.tanh(self.he(hy_0)))
        return ht, ct


class Predicter(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        dropout_rate = 0.25
        n_layers = 1
        super().__init__(
            wc=L.Linear(hidden_size * 2, hidden_size),
            he=L.Linear(hidden_size, embed_size),
            ey=L.Linear(embed_size, vocab_size),
        )

    def __call__(self, ct, ht):
        ct = F.reshape(ct, (ct.shape[1], ct.shape[2]))
        ht = F.reshape(ht, ct.shape)
        ht_tilde = F.tanh(self.wc(F.concat((ct, ht), axis=1)))
        y = self.ey(F.tanh(self.he(ht_tilde)))
        return y


class seq2seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, mecabTagger, vocab, id2wd, flag_gpu=1):
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
            decoder=Decoder(vocab_size, embed_size, hidden_size, flag_gpu),
            predicter=Predicter(vocab_size, embed_size, hidden_size),
        )
        self.mecabTagger = mecabTagger
        self.vocab = vocab
        self.id2wd = id2wd
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        if flag_gpu:
            self.ARR = cp
        else:
            self.ARR = np

        self.eos_bs = [Variable(self.ARR.array([self.vocab["<EOS>"]], dtype='int32'))
             for _ in range(self.batch_size)]
        self.eos_1 = [Variable(self.ARR.array([self.vocab["<EOS>"]], dtype='int32'))]

    def encode(self, words):
        '''
        Encoderの計算
        :param words : 入力で使用する単語(ID)のリスト
        :return :
        '''
        # エンコード
        h, c = self.encoder(words)

        # Encoderの中間ベクトル
        self.hs = h
        # 内部メモリと中間ベクトルの初期化
        # self.c = Variable(self.ARR.zeros(
        #(1, batch_size, self.hidden_size), dtype='float32'))
        # self.h = Variable(self.ARR.zeros(
        #(1, batch_size, self.hidden_size), dtype='float32'))
        # self.c = None

    def decode(self, w, x_list):
        '''
        Decoderの計算
        :param w : 入力する単語
        :return : 予測単語
        '''
        self.ht, self.c = self.decoder(w, self.ht, self.c)
        # AttentionでEncoderの加重平均を計算
        c_t = self.attention(self.hs, self.ht, x_list)
        # 予測
        t = self.predicter(c_t, self.ht)
        return t

    def reset(self, train=True):
        '''
        インスタンス変数の初期化
        :return :
        '''
        if train:
            batch_size = self.batch_size
        else:
            batch_size = 1
        self.c = Variable(self.ARR.zeros(
            (1, batch_size, self.hidden_size), dtype='float32'))
        self.ht = Variable(self.ARR.zeros(
            (1, batch_size, self.hidden_size), dtype='float32'))
        self.cleargrads()

    def __call__(self, batch):
        '''
        Lossの計算
        :param : batch 入力ミニバッチ
        :return : loss
        '''
        # 勾配の初期化
        self.reset()

        abstracts, titles = make_minibatch(
            batch, self.mecabTagger, self.vocab, self.ARR)
        # np or cp に変更
        x_list = [self.ARR.array(x, dtype='int32') for x in abstracts]
        # t_list = [self.ARR.array(x, dtype='int32') for x in titles]
        # encode
        self.encode(x_list)
        # lossの初期化
        loss = Variable(self.ARR.zeros((), dtype='float32'))
        # <EOS>をデコーダーに読み込ませる準備
        # t = Variable(self.ARR.array([self.vocab["<EOS>"] for _ in range(self.batch_size)], dtype='int32'))
        #t = [Variable(self.ARR.array([self.vocab["<EOS>"]], dtype='int32'))
        #     for _ in range(self.batch_size)]
        t = self.eos_bs
        # Decoderの計算
        # for w in t_list:
        # wを入力の形に整形
        # t = Variable([self.ARR.array(w_i, dtype='int32') for w_i in titles[0]])
        for w in titles:
            # decode
            y = self.decode(t, x_list)
            # 正解をVariableに変更
            t = Variable(self.ARR.array(w, dtype='int32'))
            # lossを計算
            loss += F.softmax_cross_entropy(y, t)
            # wを入力の形に整形
            t = [Variable(self.ARR.array([w_i], dtype='int32')) for w_i in w]
        return loss

    def predict(self, batch, interactive=False):
        '''
        予測
        :param : batch 入力
        :return : loss
        '''
        # 勾配の初期化
        self.reset(False)

        if not interactive:
            abstracts, titles = make_minibatch(
                batch, self.mecabTagger, self.vocab, self.ARR)
        else:
            abstracts = batch
        # np or cp に変更
        x_list = [self.ARR.array(x, dtype='int32') for x in abstracts]
        # t_list = [self.ARR.array(x, dtype='int32') for x in titles]
        # encode
        self.encode(x_list)
        # <EOS>をデコーダーに読み込ませる準備
        #t = [Variable(self.ARR.array([self.vocab["<EOS>"]], dtype='int32'))]
        t = self.eos_1
        # Decoderの計算
        counter = 0
        pred_title = ''
        while counter < 30:
            # decode
            y = self.decode(t, x_list)
            label = y.data[0].argmax()
            # yを入力の形に整形
            t = [Variable(self.ARR.array([label], dtype='int32'))]
            # argmaxでidを取りwordに変換
            words = self.id2wd[int(label)]
            pred_title += words
            # カウンターを進める
            counter += 1
            if label == self.vocab["<EOS>"]:
                counter = 30
        return pred_title
