import json
import collections
import pickle
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


def load_pickle(path):
    print('Loading: ', path, end='')
    start = time.time()
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(' {:.2f}'.format(time.time() - start))
    return data


def load_json(path):
    if os.path.exists(path):
        start = time.time()
        print('Load: ', path, end='')
        with open(path, 'r') as f:
            data = json.load(f)
        print(' {:d} items in  {:.2f} seconds'.format(len(data), time.time() - start))
        return data
    else:
        print('[Warning] File not found: ', path)
        return []


def save_json(data, path):
    print("Save: ", path)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def hist(populations, key=0, min=0, print_stat=True):
    counter = collections.Counter()
    counter.update(populations)
    stats = [(k, v) for k, v in counter.items() if v >= min]
    stats = sorted(stats, key=lambda x: x[key])
    if print_stat:
        for k, v in stats:
            print('{:4d}\t{}'.format(v, k))

    print('Total: ', len(stats))
    return stats


class RunningTime(object):

    def __init__(self, agent=None):
        super(RunningTime, self).__init__()
        self.agent = agent

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            start = time.time()
            f(*args, **kwargs)
            t = time.time() - start
            if self.agent:
                print("Running time of {}: {} seconds".format(self.agent, t))
            else:
                print("Running time: {} seconds".format(t))

        return wrapped_f


@RunningTime()
@RunningTime('run')
def run():
    time.sleep(2)


if __name__ == '__main__':
    run()
