import chainer
from chainer import reporter
from chainer.training import extensions
import numpy as np


class Evaluator(extensions.Evaluator):
    def __init__(self, iter, model):
        self._iterators = iter
        self._targets = {'main': model}

    def evaluate(self):
        iter = self._iterators
        model = self._targets['main']

        l = iter.dataset.__len__()
        i = 1
        summary = reporter.DictSummary()
        for batch in iter:
            observation = {}
            with reporter.report_scope(observation):
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    loss = model(batch)
                observation['val/loss'] = loss
            summary.add(observation)
            if i < l:
                i += 1
            else:
                break
        return summary.compute_mean()
