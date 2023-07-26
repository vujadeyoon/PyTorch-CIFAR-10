import json
import yaml
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from vujade import vujade_utils as utils_
from vujade.vujade_debug import printd


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_yaml(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        return yaml.dump(content, handle)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1, is_add_scalar: bool = True):
        if (self.writer is not None) and (is_add_scalar is True):
            self.add_scalar(key=key, value=value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def add_scalar(self, key, value):
        self.writer.add_scalar(key, value)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
