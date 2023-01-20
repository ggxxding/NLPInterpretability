import numpy as np
from datasets import load_dataset


class Loader(object):
    def __init__(self, data_dir, data_file = 'train',batch_size = 32, device = 'cpu'):
        self.dataset = load_dataset(data_dir,split = data_file)
        self.batch_size = batch_size
    def data_iter(self, shuffle = False):
        if (shuffle == True):
            self.dataset = self.dataset.shuffle(seed= 42 )
        data_len = len(self.dataset)
        start = 0
        while start + self.batch_size < data_len:
            yield self.dataset[start:start+self.batch_size]
            start += self.batch_size




if __name__ == '__main__':
    a=Loader('imdb','train',32)
    for i in a.data_iter(shuffle = True):
        print(i)