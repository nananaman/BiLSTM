import chainer
from chainer import reporter
from chainer.training import extensions
import numpy as np


class Evaluator(extensions.Evaluator):
    def __init__(self, iter, model, comm=None):
        self._iterators = iter
        self._targets = {'main': model}
        self.comm = comm

    def evaluate(self):
        iter = self._iterators
        model = self._targets['main']

        l = iter.dataset.__len__()
        i = 1
        summary = reporter.DictSummary()
        observation = {}
        for batch in iter:
            with reporter.report_scope(observation):
                with chainer.using_config('train', False), chainer.function.no_backprop_mode():
                    loss = model(batch)
                observation['val/loss'] = loss
            summary.add(observation)
            if i < l:
                i += iter.batch_size
            else:
                break
        return summary.compute_mean()
