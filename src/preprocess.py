import chainer
import numpy as np
import os
import random

class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, articles, lavel):
        self.articles = articles
        self.lavel = lavel

    def __len__(self):
        return len(self.lavel)

    def get_example(self, i):
        abstract = self.articles[i]['abstract']
        title = self.articles[i]['title']
        return abstract, title
