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

from preprocess import PreprocessedDataset
from updater import Updater
from visualize import out_generated_title
from evaluator import Evaluator
from BiLSTM import seq2seq
from make_easyJap_dic import make_easyJap_dic

def main():
    # データ読み込み
    PATH = "./data/T15-2018.2.28.xlsx"
    mecabTagger, articles, vocab, id2wd, la, vocab_size = make_easyJap_dic(PATH)
    print("len_articles : {0}".format(la))
    print("vocab_size : {0}".format(vocab_size))

    # データセット準備
    if Path('lavel.npy').exists():
        random_lavel = np.load('lavel.npy')
    else:
        random_lavel = np.random.permutation(np.arange(la))
        np.save('lavel', random_lavel)
    train_val, test = np.split(random_lavel, [la - 100])
    train, valid = np.split(train_val, [int(la * 0.8)])
    '''
    print('Training dataset size: ', len(train))
    print('Validation dataset size: ', len(valid))
    print('Testing dataset size: ', len(test))
    '''

    # イテレータ作成
    BATCH_SIZE = 40

    train_d = PreprocessedDataset(articles, train)
    valid_d = PreprocessedDataset(articles, valid)
    test_d = PreprocessedDataset(articles, test)
    train_iter = iterators.SerialIterator(train_d, BATCH_SIZE)
    valid_iter = iterators.SerialIterator(valid_d, BATCH_SIZE, shuffle=False)
    test_iter = iterators.SerialIterator(test_d, 1, shuffle=False)

    EMBED_SIZE = 300
    HIDDEN_SIZE = 150
    model = seq2seq(vocab_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE, mecabTagger, vocab, id2wd)
    model.reset()
    FLAG_GPU = 1
    if FLAG_GPU:
        model.to_gpu(0)

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    # Set up a trainer
    updater = Updater(
        model=(model),
        iterator={
            'main': train_iter
        },
        optimizer={
            'model': optimizer
        },
    )
    EPOCH_NUM = 50
    trainer = training.Trainer(updater, (EPOCH_NUM, 'epoch'), out='result')
    snapshot_interval = (5, 'epoch')
    display_interval = (1, 'epoch')
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(Evaluator(valid_iter, model),
                   name='val', trigger=display_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'model/loss', 'val/loss', 'elapsed_time']), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PlotReport(
        ['model/loss', 'val/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(
        out_generated_title(
            test_iter, model
        ), trigger=display_interval)

    # モデルロード
    MODEL_PATH = './result/snapshot_epoch-45'
    chainer.serializers.load_npz(MODEL_PATH, trainer)

    print('Please input a sentence!')
    S = input()
    while(S != 'c'):
        node = mecabTagger.parseToNode(S)
        node = node.next

        S_split = []
        S_split_id = []
        while node:
            word = node.surface
            node = node.next
            S_split.append(word)
            if word not in vocab:
                print('ERROR! {} is not in my vocabulary'.format(word))
            else:
                wid = vocab[word]
                S_split_id.append(wid)
        S_split_id = [S_split_id]
        
        # predict
        S_predict = model.predict(S_split_id, interactive=True)

        #print(S_split)
        print(S_predict)
        print('Please input a sentence!')
        S = input()
                

    #trainer.run()


if __name__ == '__main__':
    main()
