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
import chainermn

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

    EMBED_SIZE = 300
    HIDDEN_SIZE = 150
    BATCH_SIZE = 40

    model = seq2seq(vocab_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE, mecabTagger, vocab, id2wd)
    model.reset()
    FLAG_GPU = 1
    if FLAG_GPU:
        comm = chainermn.create_communicator()
        device = comm.intra_rank
        for i in range(comm.size):
            if comm.rank == i:
                print("Rank {} GPU: {}".format(comm.rank, device))
        chainer.backends.cuda.get_device_from_id(device).use()
        model.to_gpu(device)
    optimizer = optimizers.Adam()
    optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    if comm.rank == 0:
        print("len_articles : {0}".format(la))
        print("vocab_size : {0}".format(vocab_size))

    # データセット準備
    if Path('lavel.npy').exists():
        random_lavel = np.load('lavel.npy')
    else:
        random_lavel = np.random.permutation(np.arange(la))
        np.save('lavel', random_lavel)
    train_val, test = np.split(random_lavel, [la - 50])
    train, valid = np.split(train_val, [int(la * 0.8)])
    if comm.rank == 0:
        print('Training dataset size: ', len(train))
        print('Validation dataset size: ', len(valid))
        print('Testing dataset size: ', len(test))

    # イテレータ作成
    if comm.rank == 0:
        train_d = PreprocessedDataset(articles, train)
        valid_d = PreprocessedDataset(articles, valid)
        test_d = PreprocessedDataset(articles, test)
    else:
        train_d = None
        valid_d = None
        test_d = None
    train_d = chainermn.scatter_dataset(train_d, comm)
    valid_d = chainermn.scatter_dataset(valid_d, comm)
    test_d = chainermn.scatter_dataset(test_d, comm)
    train_iter = iterators.SerialIterator(train_d, BATCH_SIZE)
    valid_iter = iterators.SerialIterator(valid_d, BATCH_SIZE, shuffle=False)
    test_iter = iterators.SerialIterator(test_d, 1, shuffle=False)

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
    trainer.extend(chainermn.create_multi_node_evaluator(Evaluator(valid_iter, model), comm),
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

    trainer.run()

if __name__ == '__main__':
    main()
