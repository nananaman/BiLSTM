import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('model')
        self.iteration = 0
        super().__init__(*args, **kwargs)

    def update_core(self):
        optimizer = self.get_optimizer('model')
        
        batch = self.get_iterator('main').next()
        optimizer.target.zerograds()
        loss = self.model(batch)
        chainer.report({'loss': loss}, self.model)
        loss.backward()
        optimizer.update()
        
